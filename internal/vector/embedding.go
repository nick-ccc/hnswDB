package vector

import (
	"reflect"

	"github.com/nick-ccc/hnswDB/internal"
)

// EmbeddingSpace is a generic base struct representing an embedding vector space.
type EmbeddingSpace[T internal.Number] struct {
	dimensionality uint64
	sizeVector     uint64
	distanceFunc   internal.DistanceFunc[T]
}

func (s *EmbeddingSpace[T]) GetDimensionality() uint64 {
	return s.dimensionality
}

func (s *EmbeddingSpace[T]) GetDistanceFunc() internal.DistanceFunc[T] {
	return s.distanceFunc
}

func (s *EmbeddingSpace[T]) DataSize() uint64 {
	return s.sizeVector
}

func NewInnerProductSpace[T internal.Number](dim uint64) *EmbeddingSpace[T] {
	var zero T
	return &EmbeddingSpace[T]{
		dimensionality: dim,
		sizeVector:     dim * uint64(reflect.TypeOf(zero).Size()),
		distanceFunc:   EuclideanDistance[T],
	}
}

func NewCosineSimilaritySpace[T internal.Number](dim uint64) *EmbeddingSpace[T] {
	var zero T
	return &EmbeddingSpace[T]{
		dimensionality: dim,
		sizeVector:     dim * uint64(reflect.TypeOf(zero).Size()),
		distanceFunc:   CosineSimilarity[T],
	}
}
