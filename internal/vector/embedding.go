package vector

import (
	"reflect"

	"github.com/nick-ccc/hnswDB/internal"
)

// EmbeddingSpace is a generic base struct representing an embedding vector space.
type EmbeddingSpace[T internal.Number] struct {
	Dimensionality uint64
	SizeVector     uint64
	DistanceFunc   internal.DistanceFunc[T]
}

func NewEuclideanSpace[T internal.Number](dim uint64) *EmbeddingSpace[T] {
	var zero T
	return &EmbeddingSpace[T]{
		Dimensionality: dim,
		SizeVector:     dim * uint64(reflect.TypeOf(zero).Size()),
		DistanceFunc:   EuclideanDistance[T],
	}
}

func NewCosineSimilaritySpace[T internal.Number](dim uint64) *EmbeddingSpace[T] {
	var zero T
	return &EmbeddingSpace[T]{
		Dimensionality: dim,
		SizeVector:     dim * uint64(reflect.TypeOf(zero).Size()),
		DistanceFunc:   CosineSimilarity[T],
	}
}
