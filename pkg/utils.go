package pkg

// Types
type Number interface {
	~int | ~int32 | ~int64 | ~float32 | ~float64
}
type DistanceFunc[T Number] func(vector_a, vector_b []T) (float64, error)
type LabelType string
