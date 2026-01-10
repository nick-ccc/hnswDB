package vector

import (
	"reflect"

	"github.com/nick-ccc/hnswDB/pkg"
)

// EmbeddingSpace is a generic base struct representing an embedding vector space.
type EmbeddingSpace[T pkg.Number] struct {
	Dimensionality uint64
	SizeVector     uint64
	DistanceFunc   pkg.DistanceFunc[T]
}

func NewEuclideanSpace[T pkg.Number](dim uint64) *EmbeddingSpace[T] {
	var zero T
	return &EmbeddingSpace[T]{
		Dimensionality: dim,
		SizeVector:     dim * uint64(reflect.TypeOf(zero).Size()),
		DistanceFunc:   EuclideanDistance[T],
	}
}

func NewCosineSimilaritySpace[T pkg.Number](dim uint64) *EmbeddingSpace[T] {
	var zero T
	return &EmbeddingSpace[T]{
		Dimensionality: dim,
		SizeVector:     dim * uint64(reflect.TypeOf(zero).Size()),
		DistanceFunc:   CosineSimilarity[T],
	}
}
