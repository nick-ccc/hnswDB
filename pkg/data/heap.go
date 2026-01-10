package data

import "github.com/nick-ccc/hnswDB/pkg"

type Candidate struct {
	Dist float64
	Key  pkg.LabelType
}

// Min-heap (by Dist)
type MinHeapSearch []Candidate

func (h MinHeapSearch) Len() int            { return len(h) }
func (h MinHeapSearch) Less(i, j int) bool  { return h[i].Dist < h[j].Dist }
func (h MinHeapSearch) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MinHeapSearch) Push(x interface{}) { *h = append(*h, x.(Candidate)) }
func (h *MinHeapSearch) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
func (h MinHeapSearch) Max() Candidate {
	max := h[0]
	for _, c := range h {
		if c.Dist > max.Dist {
			max = c
		}
	}
	return max
}

// Max-heap (by Dist)
type MaxHeapSearch []Candidate

func (h MaxHeapSearch) Len() int            { return len(h) }
func (h MaxHeapSearch) Less(i, j int) bool  { return h[i].Dist > h[j].Dist }
func (h MaxHeapSearch) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MaxHeapSearch) Push(x interface{}) { *h = append(*h, x.(Candidate)) }
func (h *MaxHeapSearch) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func (h MaxHeapSearch) Min() Candidate {
	min := h[0]
	for _, c := range h {
		if c.Dist < min.Dist {
			min = c
		}
	}
	return min
}
