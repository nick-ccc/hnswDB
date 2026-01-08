# Low Level storage

### List layers

/**
 * HNSW Graph Node Structure
 *
 * The HNSW graph stores all nodes in a contiguous slice `Nodes []NodeLinks`, where each
 * index corresponds to the node's internal ID (TableInt). This allows O(1) access to a node
 * given its ID.
 *
 * Each `NodeLinks` represents all layers of a node in the hierarchical graph:
 *   - `Layers[0]` is the base layer, containing the most neighbors.
 *   - `Layers[1], Layers[2], ...` are upper layers, which are progressively sparser.
 *
 * Each `Layer` is represented by a `LinkList`:
 *   - `Neighbors []TableInt` stores the IDs of neighboring nodes at that layer.
 *   - The slice length gives the number of neighbors at that layer.
 *
 * Access Pattern:
 *   - To get a node by ID: `node := h.Nodes[nodeID]`
 *   - To get neighbors at a specific layer: `neighbors := node.Layers[layer].Neighbors`
 *   - To iterate neighbors: `for _, nbrID := range neighbors { ... }`
 *
 * This design mirrors the C++ HNSW memory layout but in a Go-friendly, type-safe way:
 *   - Avoids raw pointer arithmetic and unsafe memory access.
 *   - Each node stores slices of neighbors for each layer.
 *   - Layer 0 is the dense base layer; upper layers are sparser for hierarchical search.
 *
 * Example:
 *   Nodes[0].Layers[0].Neighbors = [1, 2, 5]  // base layer neighbors of node 0
 *   Nodes[0].Layers[1].Neighbors = [2]        // upper layer neighbors of node 0
 *
 * Summary:
 *   Nodes[nodeID].Layers[layer].Neighbors gives all neighbors of nodeID at that layer.
 */


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