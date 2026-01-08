package data

import (
	"container/heap"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/nick-ccc/hnswDB/internal"
	"github.com/nick-ccc/hnswDB/internal/vector"
)

type HNSWNodeID uint32
type LayerInt int32

type node[T internal.Number] struct {
	Key          internal.LabelType
	Vector       []T
	HighestLayer LayerInt
}

func makeNode[T internal.Number](
	key internal.LabelType,
	vec []T,
	maxLayer LayerInt,
	ascent_probability float64,
) node[T] {
	var level LayerInt
	for rand.Float64() < ascent_probability && level < maxLayer {
		level++
	}
	return node[T]{
		Key:          key,
		Vector:       vec,
		HighestLayer: level,
	}
}

// List of closet nodes, per node, per layer
type nodeAdjacency[T internal.Number] struct {
	Neighbors []node[T]
}

// Complete list of all layers that node makes connections to
type nodeLayerAdjacency[T internal.Number] struct {
	Node   node[T]
	Layers []nodeAdjacency[T]
}

type HNSW[T internal.Number] struct {
	// private
	entryPoint     *node[T]
	efConstruction uint16 // Number of candidate ANN during build
	efSearch       uint16 // max limit to search candidates
	maxLayer       LayerInt

	// All nodes
	nodes      []nodeLayerAdjacency[T]
	nodeLookup map[internal.LabelType]HNSWNodeID
	mutex      sync.Mutex

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

func (h *HNSW[T]) getNodeLayersFull(
	label internal.LabelType,
) *nodeLayerAdjacency[T] {
	nodeID, exist := h.getNodeIDFromLabel(label)
	if exist {
		return &h.nodes[nodeID]
	}
	return nil
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
) *node[T] {
	nodeID, exist := h.getNodeIDFromLabel(label)
	if exist {
		return &h.nodes[nodeID].Node
	}
	return nil
}

// Search later provides base search mechanics
func (h *HNSW[T]) searchLayer(
	inputVector []T,
	layer LayerInt,
	entryPoint node[T],
) (MaxHeapSearch, error) {

	// Map of visited labels
	visitedMap := make(map[internal.LabelType]struct{})

	// Init heaps for ANN search
	currentCandidates := make(MinHeapSearch, 0)
	topCandidates := make(MaxHeapSearch, 0)
	heap.Init(&currentCandidates)
	heap.Init(&topCandidates)

	// Maintain lower bound for search exploration
	lowerBound, error := h.EmbeddingSpace.DistanceFunc(inputVector, h.entryPoint.Vector)
	if error != nil {
		return nil, error
	}

	// Generate initial candidate and push to heaps
	currCandidate := Candidate{Dist: lowerBound, Key: h.entryPoint.Key}
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
		neighbors := append([]node[T]{}, currentNodeAdjacency.Neighbors...)
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

func (h *HNSW[T]) Search(
	inputVector []T,
	k int,
) []node[T] {
	var topCandidates MaxHeapSearch
	var error error
	entryPoint := h.entryPoint

	for currentLayer := h.maxLayer - 1; currentLayer >= 0; currentLayer-- {
		topCandidates, error = h.searchLayer(inputVector, currentLayer, *entryPoint)
		if error != nil {
			return nil
		}

		entryCandidate := topCandidates.Min()
		entryPoint = h.getNode(entryCandidate.Key)
	}

	sort.Slice(topCandidates, func(i, j int) bool {
		return topCandidates[i].Key < topCandidates[j].Key
	})

	result := make([]node[T], 0, min(k, int(h.efConstruction)))
	for _, x := range topCandidates {
		result = append(result, *h.getNode(x.Key))
	}

	return result
}
