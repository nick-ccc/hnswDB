package data

import (
	"github.com/nick-ccc/hnswDB/pkg"
)

type Pair[T float64, label pkg.LabelType] struct {
	Distance T
	Label    label
}

type VectorDatabase[T pkg.Number] interface {
	AddVector(
		datapoint []T,
		label pkg.LabelType,
		replaceDeleted bool,
	)

	SearchKNNUnordered(
		query any,
		k int,
	) []Pair[float64, pkg.LabelType]

	SearchKNNOrdered(
		query any,
		k int,
	) []Pair[float64, pkg.LabelType]

	SaveIndex(location string) error
}
