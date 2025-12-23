package vector

import (
	"fmt"
)

func EuclideanDistance(vector_a, vector_b any) (float32, error) {
	a := vector_a.([]float32)
	b := vector_b.([]float32)

	if len(a) != len(b) {
		return -1, fmt.Errorf(
			fmt.Sprintf(
				"Arrays must have same length, a: %d, b %d", len(a), len(b),
			),
		)
	}

	var result float32

	for idx := range a {
		result += a[idx] * b[idx]
	}
	return result, nil
}
