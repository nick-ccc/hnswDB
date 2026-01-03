package algorithim

import (
	"github.com/nick-ccc/hnswDB/internal"
)

type Pair[T float64, label internal.LabelType] struct {
	Distance T
	Label    label
}

type AlgorithmInterface[T internal.Number] interface {
	AddVector(
		datapoint []T,
		label internal.LabelType,
		replaceDeleted bool,
	)

	SearchKNNUnordered(
		query any,
		k int,
	) []Pair[float64, internal.LabelType]

	SearchKNNOrdered(
		query any,
		k int,
	) []Pair[float64, internal.LabelType]

	SaveIndex(location string) error
}
