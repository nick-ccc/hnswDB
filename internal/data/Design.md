# Search Algorithm  

### Why Use Max Heap For Top Candidates
```
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
```
- Ensures worst result is always O(1) search away (i.e. first element in the Queue)
- We want to track and compare to the worst of the best during the algorithm. Consider a case when searching and the closest unexplored candidate is worse than our current worst element, we can stop searching. It provides a lower bound for how we do ANN.