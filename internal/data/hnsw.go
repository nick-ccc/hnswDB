package data

import (
	"sync"

	"github.com/nick-ccc/hnswDB/internal"
)

// shoulder layer be its own struct???
type node[T internal.Number] struct {
	key          internal.LabelType
	value        []T
	highestLayer uint16
	// map of pointers to other Nodes
	neighbors map[internal.LabelType]*node[T]
}

func makeNode[T internal.Number](
	key internal.LabelType,
	vec []T,
	maxLayer uint16,
	ascent_probability float64,
) node[T] {
	var level uint16
	for rand.Float64() < prob && level < maxLevel {
        level++
    }

	return node[T]{
		key:          key,
		value:        vec,
		highestLayer: level,
		neighbors:    nil,
	}
}

type LayeredGraph[T internal.Number] struct {
	maxConnections uint16
	elementLevels  []uint16
}

type HNSW[T internal.Number] struct {
	// Specific to graph layers
	entryPoint     *node[T]
	maxNeighbors   uint16 // Max neighbors per node
	efConstruction uint16 // Number of candidate NN during build
	nodes          map[internal.LabelType]*node[T]

	maxElements     uint64
	curElementCount uint64
	dataDimensions  uint64
	distFunc        internal.DistanceFunc[T]
	indexLock       sync.Mutex
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