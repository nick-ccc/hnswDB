package data

import (
	"container/heap"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/nick-ccc/hnswDB/internal"
	"github.com/nick-ccc/hnswDB/internal/vector"
)

type HNSWNodeID uint32
type LayerInt int32

// Node
type Node[T internal.Number] struct {
	Key          internal.LabelType
	Vector       []T
	HighestLayer LayerInt
}

// Used for inputting vectors top become nodes
type InputPairing[T internal.Number] struct {
	Vector []T
	Key    internal.LabelType
}

func makeNode[T internal.Number](
	key internal.LabelType,
	vec []T,
	maxNumLayers LayerInt,
	ascent_probability float64,
) Node[T] {
	var level LayerInt
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
type nodeAdjacency[T internal.Number] struct {
	Neighbors []*Node[T]
}

// Complete list of all layers that node makes connections to
type nodeLayerAdjacency[T internal.Number] struct {
	Node   *Node[T]
	Layers []nodeAdjacency[T]
}

type HNSW[T internal.Number] struct {
	// private
	entryPoint     *Node[T]
	efConstruction uint16 // Number of candidate ANN during build
	efSearch       uint16 // max limit to search candidates
	maxNumLayers   LayerInt
	ascentProb     float64

	// All nodes
	nodes              []*nodeLayerAdjacency[T] // each idx corresponds to matching ID
	nodeLookup         map[internal.LabelType]HNSWNodeID
	reverseLayerLookup map[LayerInt][]HNSWNodeID
	mutex              sync.Mutex

	// public
	MaxElements     HNSWNodeID
	CurElementCount HNSWNodeID
	EmbeddingSpace  vector.EmbeddingSpace[T]
	MaxNeighbors    uint16 // Max neighbors per node
}

func (h *HNSW[T]) getNodeIDFromLabel(
	label internal.LabelType,
) (HNSWNodeID, bool) {
	if nodeID, exist := h.nodeLookup[label]; exist {
		return nodeID, true
	}
	return 0, false
}

func (h *HNSW[T]) getNodeLayerAdjacency(
	label internal.LabelType,
	layer LayerInt,
) *nodeAdjacency[T] {
	nodeID, exist := h.getNodeIDFromLabel(label)
	if exist {
		return &h.nodes[nodeID].Layers[layer]
	}
	return nil
}

func (h *HNSW[T]) getNode(
	label internal.LabelType,
) *Node[T] {
	nodeID, exist := h.getNodeIDFromLabel(label)
	if exist {
		return h.nodes[nodeID].Node
	}
	return nil
}

func (h *HNSW[T]) getRandomEntryFromLayer(
	layer LayerInt,
) *nodeAdjacency[T] {
	if _, exists := h.reverseLayerLookup[layer]; exists {
		return nil
	}

	if len(h.reverseLayerLookup[layer]) > 0 {
		randID := rand.IntN(len(h.reverseLayerLookup[layer]))
		return &h.nodes[randID].Layers[layer]
	}

	return nil
}

// Search layer provides base search mechanics
func (h *HNSW[T]) searchLayer(
	inputVector []T,
	layer LayerInt,
	entryPoint Node[T],
) (MaxHeapSearch, error) {

	// Map of visited labels
	visitedMap := make(map[internal.LabelType]struct{})

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
		if currCandidate.Dist > lowerBound && topCandidates.Len() == int(h.efConstruction) {
			// Exit condition if distance of next candidate is smaller than what is
			// currently in ANN heap, and heap is full (equal to ef construction)
			break
		}
		heap.Pop(&currentCandidates)
		currentNodeAdjacency := h.getNodeLayerAdjacency(currCandidate.Key, layer)

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
			if topCandidates.Len() < int(h.efConstruction) || lowerBound > distance {
				// either top candidate list is not full or new distance is smallest yet
				lowerBound = topCandidates[0].Dist
				newCandidate := Candidate{Dist: distance, Key: neighborNode.Key}
				heap.Push(&topCandidates, newCandidate)
				heap.Push(&currentCandidates, newCandidate)

				// Keep top candidates size less than or equal to efConstruction
				if topCandidates.Len() > int(h.efConstruction) {
					heap.Pop(&topCandidates)
				}

				if currentCandidates.Len() > int(h.efSearch) {
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

	for currentLayer := h.maxNumLayers - 1; currentLayer >= 0; currentLayer-- {
		topCandidates, error = h.searchLayer(inputVector, currentLayer, *entryPoint)
		if error != nil {
			return nil, error
		}

		entryCandidate := topCandidates.Min()
		entryPoint = h.getNode(entryCandidate.Key)
	}

	sort.Slice(topCandidates, func(i, j int) bool {
		return topCandidates[i].Key < topCandidates[j].Key
	})

	result := make([]Node[T], 0, min(k, int(h.efConstruction)))
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

	invalidIndices := []int{}
	for idx, item := range inputPairs {
		if len(item.Vector) != int(h.EmbeddingSpace.Dimensionality) {
			invalidIndices = append(invalidIndices, idx)
		} else if _, exists := h.nodeLookup[item.Key]; exists {
			slog.Warn("Key replacement not implemented yet, skipping...")
			continue
		}

		// generate node
		node := makeNode(
			item.Key,
			item.Vector,
			h.maxNumLayers,
			h.ascentProb,
		)
		nodeLayered := nodeLayerAdjacency[T]{
			Node:   &node,
			Layers: []nodeAdjacency[T]{},
		}

		// @Note node.HighestLayer is already 0 indexed
		for i := node.HighestLayer; i >= 0; i-- {
			entryAdjacencyList := h.getRandomEntryFromLayer(i)

			if entryAdjacencyList == nil {
				// It is the only node in the layer
				continue
			}

		}

		// Lock mutex while generating unique ID
		// Add ID to lookup table
		h.mutex.Lock()
		h.CurElementCount++
		id := h.CurElementCount
		h.nodeLookup[item.Key] = id
		for i := node.HighestLayer; i >= 0; i-- {
			h.reverseLayerLookup[node.HighestLayer] = append(
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
