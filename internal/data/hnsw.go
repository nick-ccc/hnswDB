package data

import (
	"sync"

	"github.com/nick-ccc/hnswDB/internal"
)

type HNSWNodeID uint32
type LayerInt uint16

type node[T internal.Number] struct {
	ID			 HNSWNodeID
	Key          internal.LabelType
	Value        []T
	HighestLayer LayerInt
}

func makeNode[T internal.Number](
	key internal.LabelType,
	vec []T,
	maxLayer LayerInt,
	ascent_probability float64,
) node[T] {
	var level LayerInt
	for rand.Float64() < prob && level < maxLevel {
        level++
    }
	return node[T]{
		Key:          key,
		Value:        vec,
		HighestLayer: level,
	}
}

type nodeAdjacency[T internal.Number] struct {
	Neighbors []node[T]
}

type layerAdjacency[T internal.Number] struct {
	Layers []nodeAdjacency[T]
}

type HNSW[T internal.Number] struct {
	// private
	entryPoint     	*node[T]
	efConstruction 	uint16 // Number of candidate ANN during build
	nodes          	[]layerAdjacency[T]
	mutex       	sync.Mutex

	// public
	MaxElements     uint64
	CurElementCount uint64
	EmbeddingSpace  EmbeddingSpace[T]
	MaxNeighbors    uint16 // Max neighbors per node
}

func (h *HNSW[T]) getNeighborsAtLevel(
	nodeID HNSWNodeID,
	layer LayerInt,
) []node {
	if layer < 0 || int(HNSWNodeID) >= len(h.nodes) {
		return nil
	}

	node := h.nodes[nodeID]
	if layer >= len(node.HighestLayer) {
		return nil
	}

	return node.Layers[layer].Neighbors
}

// Search later provides base search mechanics
func (h *HNSW[T])searchLayer(
	data_point []T,
	layer LayerInt,
) (MaxHeapSearch, error) {

	// Map of visited labels
	visitedList = make(map[internal.LabelType]struct{})

	// Init heaps for ANN search
	currentCandidates := make(MinHeapSearch, 0)
	topCandidates := make(MaxHeapSearch, 0)
	heap.init(&currentCandidates)
	heap.init(&topCandidates)

	// Maintain lower bound for search exploration
	lowerBound := h.EmbeddingSpace.DistanceFunc(data_point, h.entryPoint)

	// Generate initial candidate and push to heaps
	currCandidate := Candidate {Dist: lowerBound, Key: h.entryPoint.key}
	heap.Push(&topCandidates, currCandidate)
	heap.Push(&currentCandidates, currCandidate)
	visitedList[h.entryPoint.key] = struct{}{}

	for len(currentCandidates) > 0 {
		currCandidate = currentCandidates[0]
		if currCandidate.Dist > lowerBound && topCandidates.Len() == h.efConstruction {
			// Exit condition if distance of next candidate is smaller than what is 
			// currently in ANN heap, and heap is full (equal to ef construction)
			break
		}
		heap.pop(&currentCandidates)
		currentCandidateKey := currCandidate.Key

		// Lock index while searching
		h.mutex.Lock()

		if layer == 0 {
			continue 
		} else {
			// Working on layout here
		}
		
		for 
		distance := h.embeddingSpace.DistanceFunc()
	}
}

func (h *HNSW[T]) addNeighbors(
	node node[T],
	layer uint16,
) error {
	return nil
}

func (h *HNSW[T]) AddVector(
	label internal.LabelType,
	vector []T,
	ascent_probability,
) error {
	// Go through each layer and add neighbors up adding the 
	// maxNeighbors with closest 
	c
	for i := 0; i < int(newNode.highestLayer); i++ {
		h.addNeighbors()
	}
}