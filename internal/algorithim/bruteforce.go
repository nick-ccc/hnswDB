package algorithim

import (
	"sync"

	"github.com/nick-ccc/hnswDB/internal"
)

type BruteforceSearch[T internal.Number] struct {
	AlgorithmInterface[T]

	data            []byte
	maxElements     uint64
	curElementCount uint64
	sizePerElement  uint64
	dataSize        uint64

	distFunc      internal.DistanceFunc[T]
	distFuncParam any

	indexLock          sync.Mutex
	externalToInternal map[internal.LabelType]uint64
}

func NewBruteforceSearch[T internal.Number](s SpaceInterface[T]) *BruteforceSearch[T] {
	return &BruteforceSearch[T]{
		data:               nil,
		maxElements:        0,
		curElementCount:    0,
		sizePerElement:     0,
		dataSize:           s.DataSize(),
		distFunc:           s.DistanceFunc(),
		distFuncParam:      s.DistanceFuncParam(),
		externalToInternal: make(map[internal.LabelType]uint64),
	}
}
