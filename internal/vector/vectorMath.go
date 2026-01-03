package vector

import (
	"fmt"
	"math"

	"github.com/nick-ccc/hnswDB/internal"
)

// Check for valid inputs
func validityCheck[T internal.Number](vector_a, vector_b []T) error {

	if len(vector_a) != len(vector_b) {
		return fmt.Errorf(
			fmt.Sprintf(
				"Arrays must have same length, a: %d, b %d", len(vector_a), len(vector_b),
			),
		)
	}

	return nil
}

// Computes dot product result of two input vectors
func dotProduct[T internal.Number](vector_a, vector_b []T) float64 {
	// assumes same length
	var result T
	for idx := range vector_a {
		result += vector_a[idx] * vector_b[idx]
	}
	return float64(result)
}

// Computes norm of input vector
func vectorNorm[T internal.Number](v []T) float64 {
	return math.Sqrt(dotProduct(v, v))
}

// EuclideanDistance computes the L2 (Euclidean) distance
// between two vectors.
func EuclideanDistance[T internal.Number](vector_a, vector_b []T) (float64, error) {
	error := validityCheck(vector_a, vector_b)
	if error != nil {
		return -1.0, error
	}

	var sum T
	for idx := range vector_a {
		diff := vector_a[idx] - vector_b[idx]
		sum += diff * diff
	}
	return math.Pow(float64(sum), 0.5), nil
}

// CosineDistance computes the cosine distance (1 - cosine similarity)
// between two vectors.
func CosineSimilarity[T internal.Number](vector_a, vector_b []T) (float64, error) {
	error := validityCheck(vector_a, vector_b)
	if error != nil {
		return -1.0, error
	}

	dot := dotProduct(vector_a, vector_b)
	normA, normB := vectorNorm(vector_a), vectorNorm(vector_b)

	if normA == 0 || normB == 0 {
		return -1.0, fmt.Errorf("cosine distance undefined for zero vector")
	}

	cosineSimilarity := dot / (normA * normB)
	return 1.0 - cosineSimilarity, nil
}
