package data

import (
	"fmt"
	"sync"

	"github.com/nick-ccc/hnswDB/internal"
	"github.com/nick-ccc/hnswDB/internal/vector"
)

type BruteforceSearch[T internal.Number] struct {
	data            []T
	maxElements     uint64
	curElementCount uint64
	dataDimensions  uint64

	distFunc internal.DistanceFunc[T]

	indexLock sync.Mutex

	// Data labels to index mapper
	externalToInternal map[internal.LabelType]uint64
}

// Creates Bruteforce Search space with given embedding space
// maxElements defines the maximum capacity of stored vectors.
func NewBruteforceSearch[T internal.Number](
	s vector.EmbeddingSpace[T],
	maxElements uint64,
) *BruteforceSearch[T] {
	return &BruteforceSearch[T]{
		data:               nil,
		maxElements:        maxElements,
		curElementCount:    0,
		dataDimensions:     s.GetDimensionality(),
		distFunc:           s.GetDistanceFunc(),
		externalToInternal: make(map[internal.LabelType]uint64),
	}
}

func (s *BruteforceSearch[T]) AddData(
	datapoint []T,
	label internal.LabelType,
	replaceDeleted bool,
) error {
	s.indexLock.Lock()
	defer s.indexLock.Unlock()

	var idx uint64
	if existingIdx, ok := s.externalToInternal[label]; ok {
		idx = existingIdx
	} else {
		if s.curElementCount >= s.maxElements {
			return fmt.Errorf(
				"the number of elements exceeds the specified limit",
			)
		}
		idx = s.curElementCount
		s.externalToInternal[label] = idx
		s.curElementCount++
	}

	if storeInMemory(s.data) != nil {

	}

	return nil
}
