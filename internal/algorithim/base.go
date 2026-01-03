package algorithim

import (
	"github.com/nick-ccc/hnswDB/internal"
)

type Pair[T float64, label internal.LabelType] struct {
	Distance T
	Label    label
}

type Algorithm[T internal.Number] interface {
	AddData(
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

func storeInMemory[T internal.Number](data []T) error {
	return nil
}
