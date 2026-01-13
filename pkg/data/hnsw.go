package data

import (
	"container/heap"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/nick-ccc/hnswDB/pkg"
	"github.com/nick-ccc/hnswDB/pkg/vector"
)

// Node
type Node[T pkg.Number] struct {
	Key          pkg.LabelType
	Vector       []T
	HighestLayer int
}

// Used for inputting vectors top become nodes
type InputPairing[T pkg.Number] struct {
	Vector []T
	Key    pkg.LabelType
}

func makeNode[T pkg.Number](
	key pkg.LabelType,
	vec []T,
	maxNumLayers int,
	ascent_probability float64,
) Node[T] {
	var level int
	for rand.Float64() < ascent_probability && level < maxNumLayers-1 {
		level++
	}
	return Node[T]{
		Key:          key,
		Vector:       vec,
		HighestLayer: level,
	}
}

// List of closet Nodes, per Node, per layer
type nodeAdjacency[T pkg.Number] struct {
	Neighbors []*Node[T]
}

// Complete list of all layers that node makes connections to
type nodeLayerAdjacency[T pkg.Number] struct {
	Node   *Node[T]
	Layers []nodeAdjacency[T]
}

func (n *nodeLayerAdjacency[T]) addNeighbor(
	newNode *Node[T],
	maxNeighbors int,
	layer int,
	dist pkg.DistanceFunc[T],
) error {
	neighbors := &n.Layers[layer].Neighbors
	if *neighbors == nil {
		*neighbors = []*Node[T]{newNode}
	}

	newNodeDistance, err := dist(n.Node.Vector, newNode.Vector)
	if err != nil {
		return fmt.Errorf("Failed to compute distance, %w", err)
	}

	// Keep neighbors in sorted order
	idx := 0
	for _, node := range *neighbors {
		currNodeDistance, err := dist(n.Node.Vector, node.Vector)
		if err != nil {
			return fmt.Errorf("Failed to compute distance, %w", err)
		}

		// Condition to end while loop
		if newNodeDistance > currNodeDistance {
			break
		}

		// iterate index
		idx++
	}

	// Keep to maxNeighbors length
	*neighbors = append(*neighbors, nil)
	copy((*neighbors)[idx+1:], (*neighbors)[idx:])
	(*neighbors)[idx] = newNode
	if len(*neighbors) > maxNeighbors {
		*neighbors = (*neighbors)[:maxNeighbors]
	}

	return nil
}

type HNSW[T pkg.Number] struct {
	// private
	entryPoint         *Node[T]
	nodes              []*nodeLayerAdjacency[T] // each idx corresponds to matching ID
	nodeLookup         map[pkg.LabelType]int
	reverseLayerLookup map[int][]int
	mutex              sync.Mutex

	// public
	EfConstruction  int // Number of candidate ANN during build
	EfSearch        int // max limit to search candidates
	MaxNumLayers    int
	AscentProb      float64
	MaxElements     int
	CurElementCount int
	EmbeddingSpace  *vector.EmbeddingSpace[T]
	MaxNeighbors    int // Max neighbors per node
}

func MakeHNSWEuclidean[T pkg.Number](
	efConstruction int,
	efSearch int,
	maxNumLayers int,
	ascentProb float64,
	maxElements int,
	dimensions uint64,
	maxNeighbors int,
) HNSW[T] {
	embeddingSpace := vector.NewEuclideanSpace[T](dimensions)

	return HNSW[T]{
		entryPoint:         nil,
		nodes:              []*nodeLayerAdjacency[T]{},
		nodeLookup:         make(map[pkg.LabelType]int),
		reverseLayerLookup: make(map[int][]int),
		mutex:              sync.Mutex{},
		EfConstruction:     efConstruction,
		EfSearch:           efSearch,
		MaxNumLayers:       maxNumLayers,
		AscentProb:         ascentProb,
		MaxElements:        maxElements,
		CurElementCount:    0,
		EmbeddingSpace:     embeddingSpace,
		MaxNeighbors:       maxNeighbors,
	}
}

func (h *HNSW[T]) getNodeIDFromLabel(
	label pkg.LabelType,
) (int, bool) {
	if nodeID, exist := h.nodeLookup[label]; exist {
		return nodeID, true
	}
	return 0, false
}

func (h *HNSW[T]) getNodeAdjacency(
	label pkg.LabelType,
	layer int,
) *nodeAdjacency[T] {
	nodeID, exist := h.getNodeIDFromLabel(label)
	if exist {
		return &h.nodes[nodeID].Layers[layer]
	}
	return nil
}

func (h *HNSW[T]) getNodeLayerAdjacency(
	label pkg.LabelType,
) *nodeLayerAdjacency[T] {
	nodeID, exist := h.getNodeIDFromLabel(label)
	if exist {
		return h.nodes[nodeID]
	}
	return nil
}

func (h *HNSW[T]) getNode(
	label pkg.LabelType,
) *Node[T] {
	nodeID, exist := h.getNodeIDFromLabel(label)
	if exist {
		return h.nodes[nodeID].Node
	}
	return nil
}

func (h *HNSW[T]) getRandomEntryFromLayer(
	layer int,
) *Node[T] {
	neighbors := &h.reverseLayerLookup
	if _, exists := (*neighbors)[layer]; !exists {
		return nil
	}

	lenNeighbors := len((*neighbors)[layer])
	if lenNeighbors > 0 {
		randIdx := rand.IntN(lenNeighbors)
		randID := (*neighbors)[layer][randIdx]
		return h.nodes[randID].Node
	}

	return nil
}

func (h *HNSW[T]) makeNode(
	key pkg.LabelType,
	vec []T,
) *Node[T] {
	if h.entryPoint != nil {
		// Generate Node at random level
		node := makeNode(
			key,
			vec,
			h.MaxNumLayers,
			h.AscentProb,
		)
		return &node
	}
	// Otherwise it is first node, assert ascent to highest layer
	// and set as naive entrypoint
	entrypoint := &Node[T]{
		Key:          key,
		Vector:       vec,
		HighestLayer: h.MaxNumLayers - 1,
	}
	h.entryPoint = entrypoint
	return entrypoint
}

// Search layer provides base search mechanics
func (h *HNSW[T]) searchLayer(
	inputVector []T,
	layer int,
	entryPoint *Node[T],
) (MaxHeapSearch, error) {

	// Map of visited labels
	visitedMap := make(map[pkg.LabelType]struct{})

	// Init heaps for ANN search
	currentCandidates := make(MinHeapSearch, 0)
	topCandidates := make(MaxHeapSearch, 0)
	heap.Init(&currentCandidates)
	heap.Init(&topCandidates)

	// Maintain lower bound for search exploration
	lowerBound, error := h.EmbeddingSpace.DistanceFunc(inputVector, entryPoint.Vector)
	if error != nil {
		return nil, error
	}

	// Generate initial candidate and push to heaps
	currCandidate := Candidate{Dist: lowerBound, Key: entryPoint.Key}
	heap.Push(&topCandidates, currCandidate)
	heap.Push(&currentCandidates, currCandidate)

	// Add to visited map
	visitedMap[h.entryPoint.Key] = struct{}{}

	for currentCandidates.Len() > 0 {
		currCandidate = currentCandidates[0]
		if currCandidate.Dist > lowerBound && topCandidates.Len() == int(h.EfConstruction) {
			// Exit condition if distance of next candidate is smaller than what is
			// currently in ANN heap, and heap is full (equal to ef construction)
			break
		}
		heap.Pop(&currentCandidates)
		currentNodeAdjacency := h.getNodeAdjacency(currCandidate.Key, layer)

		// Lock index to get current adjacency at time of search
		h.mutex.Lock()
		neighbors := append([]*Node[T]{}, currentNodeAdjacency.Neighbors...)
		h.mutex.Unlock()

		for _, neighborNode := range neighbors {
			if _, exist := visitedMap[neighborNode.Key]; exist {
				continue
			}
			visitedMap[neighborNode.Key] = struct{}{}
			distance, error := h.EmbeddingSpace.DistanceFunc(inputVector, neighborNode.Vector)
			if error != nil {
				return nil, error
			}

			// Check if current node is potential candidate
			if topCandidates.Len() < int(h.EfConstruction) || lowerBound > distance {
				// either top candidate list is not full or new distance is smallest yet
				lowerBound = topCandidates[0].Dist
				newCandidate := Candidate{Dist: distance, Key: neighborNode.Key}
				heap.Push(&topCandidates, newCandidate)
				heap.Push(&currentCandidates, newCandidate)

				// Keep top candidates size less than or equal to efConstruction
				if topCandidates.Len() > int(h.EfConstruction) {
					heap.Pop(&topCandidates)
				}

				if currentCandidates.Len() > int(h.EfSearch) {
					// Implement in future
				}
			}
		}
	}
	return topCandidates, nil
}

// Search through entire HNSW for ANN
func (h *HNSW[T]) Search(
	inputVector []T,
	k int,
) ([]Node[T], error) {
	var topCandidates MaxHeapSearch
	var error error
	entryPoint := h.entryPoint

	for currentLayer := h.MaxNumLayers - 1; currentLayer >= 0; currentLayer-- {
		topCandidates, error = h.searchLayer(inputVector, currentLayer, entryPoint)
		if error != nil {
			return nil, error
		}

		entryCandidate := topCandidates.Min()
		entryPoint = h.getNode(entryCandidate.Key)
	}

	sort.Slice(topCandidates, func(i, j int) bool {
		return topCandidates[i].Dist < topCandidates[j].Dist
	})

	result := make([]Node[T], 0, min(k, int(h.EfConstruction)))
	for _, x := range topCandidates {
		result = append(result, *h.getNode(x.Key))
	}

	return result, nil
}

// Adds node(s) to HNSW network
func (h *HNSW[T]) Add(
	inputPairs ...InputPairing[T],
) error {
	if int(h.CurElementCount)+len(inputPairs) >= int(h.MaxElements) {
		return fmt.Errorf(
			"Unable to add %d nodes, current node count is %d out of %d",
			len(inputPairs),
			h.CurElementCount,
			h.MaxElements,
		)
	}

	invalidMessages := []string{}
	for idx, item := range inputPairs {
		if len(item.Vector) != int(h.EmbeddingSpace.Dimensionality) {
			invalidMessages = append(
				invalidMessages,
				fmt.Sprintf("Invalid vector dimension from input: %d", idx),
			)
		} else if _, exists := h.nodeLookup[item.Key]; exists {
			slog.Warn("Key replacement not implemented yet, skipping...")
			continue
		}

		// generate node
		// @ note it would be better to do this in for loop below,
		// but if you start at layer 4, and index then it will error
		// could make the for loop ascending but this is inverse
		// to how searching would be done in production hnsw
		node := h.makeNode(
			item.Key,
			item.Vector,
		)
		nodeLayered := nodeLayerAdjacency[T]{
			Node:   node,
			Layers: make([]nodeAdjacency[T], node.HighestLayer+1),
		}
		for i := range nodeLayered.Layers {
			nodeLayered.Layers[i] = nodeAdjacency[T]{
				Neighbors: []*Node[T]{},
			}
		}

		// @Note node.HighestLayer is already 0 indexed
		for layer := node.HighestLayer; layer >= 0; layer-- {

			entryPoint := h.getRandomEntryFromLayer(layer)
			if entryPoint == nil {
				// It is the only node in the layer
				// thus prepend empty adjacency
				continue
			}

			potentialNeighbors, err := h.searchLayer(item.Vector, layer, entryPoint)
			if err != nil {
				invalidMessages = append(
					invalidMessages,
					fmt.Sprintf("pkg issue during search on input: %d", idx),
				)
				continue
			}

			for _, candidateNeighbor := range potentialNeighbors {
				candidateLayer := h.getNodeLayerAdjacency(candidateNeighbor.Key)
				candidateNode := candidateLayer.Node

				// modify neighbors of candidates and node
				h.mutex.Lock()
				nodeLayered.addNeighbor(
					candidateNode,
					h.MaxNeighbors,
					layer,
					h.EmbeddingSpace.DistanceFunc,
				)
				candidateLayer.addNeighbor(
					node,
					h.MaxNeighbors,
					layer,
					h.EmbeddingSpace.DistanceFunc,
				)
				h.mutex.Unlock()
			}
		}

		// Lock mutex while generating unique ID
		// Add ID to lookup table, then increment node count
		h.mutex.Lock()
		id := h.CurElementCount
		h.nodeLookup[item.Key] = id
		h.CurElementCount++
		for i := node.HighestLayer; i >= 0; i-- {
			h.reverseLayerLookup[i] = append(
				h.reverseLayerLookup[i],
				id,
			)
		}
		h.nodes = append(
			h.nodes,
			&nodeLayered,
		)
		h.mutex.Unlock()
	}

	// ToDO
	return nil
}
