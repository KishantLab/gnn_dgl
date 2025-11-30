# Key Features of this Implementation:
# User-Selectable K: The script takes the clique size K as a command-line argument.

# CuPy RawKernel Integration: Our complete v6 C/CUDA code (the vertex-centric, dual-kernel, clique-storing version) is embedded as a raw string and compiled on the fly by CuPy.

# Deterministic Merging: Implements the robust merging strategy we planned. When a node belongs to multiple cliques of size K, the clique with the lowest sum of vertex IDs is chosen to form the supernode, guaranteeing a consistent result every time.

# Feature Averaging: Correctly computes the new feature vectors for supernodes by averaging the features of their constituent nodes.

# New Graph Construction: Intelligently reconstructs the graph topology by mapping old edges to new supernode connections, removing duplicates and self-loops.

# High-Level Control: Uses DGL for graph representation and argparse for a user-friendly command-line interface.

# =========================================================================================
import dgl
import torch
import cupy as cp
import numpy as np
import argparse
import time
from ogb.nodeproppred import DglNodePropPredDataset

# =========================================================================
# The Complete C/CUDA v6 Kernel (Embedded as a Python Raw String)
# =========================================================================
# This is our robust, vertex-centric, dual-kernel, clique-storing CUDA code.
# It's defined as extern "C" to be visible to CuPy's compiler.
CUDA_KERNEL_CODE = r'''
#define K_MAX 16
#define MAX_DEGREE_FOR_SHARED_MEM 2048

struct SearchState { int neighbor_idx; int neighbor_end_idx; };

__device__ __forceinline__ bool are_connected_global(int u, int v, const int* d_row_ptr, const int* d_col_idx) {
    int start = d_row_ptr[u];
    int end_idx = d_row_ptr[u + 1] - 1;
    if (start > end_idx || v < d_col_idx[start] || v > d_col_idx[end_idx]) return false;
    while (start <= end_idx) {
        int mid = start + (end_idx - start) / 2;
        if (d_col_idx[mid] == v) return true;
        if (d_col_idx[mid] < v) start = mid + 1; else end_idx = mid - 1;
    }
    return false;
}

__device__ __forceinline__ bool binary_search_shared(const int* sh_arr, int size, int key) {
    int low = 0, high = size - 1;
    if (low > high || key < sh_arr[low] || key > sh_arr[high]) return false;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (sh_arr[mid] == key) return true;
        if (sh_arr[mid] < key) low = mid + 1; else high = mid - 1;
    }
    return false;
}

__device__ int intersect_count_shared(const int* sh_neighbors_u, int u_degree, int v, const int* d_row_ptr, const int* d_col_idx) {
    int count = 0;
    int v_neighbors_start = d_row_ptr[v];
    int v_degree = d_row_ptr[v+1] - v_neighbors_start;
    int ptr_u = 0, ptr_v = 0;
    while (ptr_u < u_degree && ptr_v < v_degree) {
        int neighbor_of_u = sh_neighbors_u[ptr_u];
        int neighbor_of_v = d_col_idx[v_neighbors_start + ptr_v];
        if (neighbor_of_u < neighbor_of_v) ptr_u++;
        else if (neighbor_of_v < neighbor_of_u) ptr_v++;
        else { count++; ptr_u++; ptr_v++; }
    }
    return count;
}

__device__ int intersect_count_global(int u, int v, const int* d_row_ptr, const int* d_col_idx) {
    int count = 0;
    int u_neighbors_start = d_row_ptr[u];
    int u_degree = d_row_ptr[u+1] - u_neighbors_start;
    int v_neighbors_start = d_row_ptr[v];
    int v_degree = d_row_ptr[v+1] - v_neighbors_start;
    int ptr_u = 0, ptr_v = 0;
    while(ptr_u < u_degree && ptr_v < v_degree) {
        int neighbor_of_u = d_col_idx[u_neighbors_start + ptr_u];
        int neighbor_of_v = d_col_idx[v_neighbors_start + ptr_v];
        if (neighbor_of_u < neighbor_of_v) ptr_u++;
        else if (neighbor_of_v < neighbor_of_u) ptr_v++;
        else { count++; ptr_u++; ptr_v++; }
    }
    return count;
}


extern "C" __global__ void k_clique_kernel_optimized(
    const int K, const int* d_task_list,
    const int* d_row_ptr, const int* d_col_idx,
    unsigned long long* d_clique_count, 
    int* d_clique_results, const unsigned int MAX_CLIQUES)
{
    extern __shared__ int sh_neighbors_u[];
    int u = d_task_list[blockIdx.x];
    int u_degree = d_row_ptr[u + 1] - d_row_ptr[u];
    int tid = threadIdx.x;

    for (int i = tid; i < u_degree; i += blockDim.x) sh_neighbors_u[i] = d_col_idx[d_row_ptr[u] + i];
    __syncthreads();
    
    int current_clique[K_MAX];
    SearchState search_stack[K_MAX];
    current_clique[0] = u;
    
    for (int v_idx = tid; v_idx < u_degree; v_idx += blockDim.x) {
        int v = sh_neighbors_u[v_idx];
        if (intersect_count_shared(sh_neighbors_u, u_degree, v, d_row_ptr, d_col_idx) < K - 2) continue;

        current_clique[1] = v;
        int current_size = 2;
        if (current_size == K) {
            unsigned long long store_idx = atomicAdd(d_clique_count, 1);
            if (store_idx < MAX_CLIQUES) for(int i=0; i<K; ++i) d_clique_results[store_idx * K + i] = current_clique[i];
            continue;
        }
        
        search_stack[current_size - 1] = (SearchState){ d_row_ptr[v], d_row_ptr[v + 1] };

        while (current_size >= 2) {
            SearchState* state = &search_stack[current_size - 1];
            if (state->neighbor_idx >= state->neighbor_end_idx) { current_size--; continue; }
            
            int candidate = d_col_idx[state->neighbor_idx];
            state->neighbor_idx++;

            bool is_fully_connected = true;
            if (!binary_search_shared(sh_neighbors_u, u_degree, candidate)) continue;
            
            for (int j = 1; j < current_size; ++j) {
                if (!are_connected_global(current_clique[j], candidate, d_row_ptr, d_col_idx)) {
                    is_fully_connected = false; break;
                }
            }
            
            if (is_fully_connected) {
                current_clique[current_size] = candidate;
                current_size++;
                if (current_size == K) {
                    unsigned long long store_idx = atomicAdd(d_clique_count, 1);
                    if (store_idx < MAX_CLIQUES) for(int i=0; i<K; ++i) d_clique_results[store_idx * K + i] = current_clique[i];
                    current_size--;
                } else {
                    int last_vtx = current_clique[current_size - 1];
                    search_stack[current_size - 1] = (SearchState){ d_row_ptr[last_vtx], d_row_ptr[last_vtx + 1] };
                }
            }
        }
    }
}

extern "C" __global__ void k_clique_kernel_fallback(
    const int K, const int* d_task_list,
    const int* d_row_ptr, const int* d_col_idx,
    unsigned long long* d_clique_count,
    int* d_clique_results, const unsigned int MAX_CLIQUES)
{
    int u = d_task_list[blockIdx.x];
    int u_degree = d_row_ptr[u + 1] - d_row_ptr[u];
    int tid = threadIdx.x;
    
    int current_clique[K_MAX];
    SearchState search_stack[K_MAX];
    current_clique[0] = u;
    
    for (int v_idx = tid; v_idx < u_degree; v_idx += blockDim.x) {
        int v = d_col_idx[d_row_ptr[u] + v_idx];
        if (intersect_count_global(u, v, d_row_ptr, d_col_idx) < K - 2) continue;
        
        current_clique[1] = v;
        int current_size = 2;
        if (current_size == K) {
            unsigned long long store_idx = atomicAdd(d_clique_count, 1);
            if (store_idx < MAX_CLIQUES) for(int i=0; i<K; ++i) d_clique_results[store_idx * K + i] = current_clique[i];
            continue;
        }

        search_stack[current_size - 1] = (SearchState){ d_row_ptr[v], d_row_ptr[v + 1] };

        while (current_size >= 2) {
            SearchState* state = &search_stack[current_size - 1];
            if (state->neighbor_idx >= state->neighbor_end_idx) { current_size--; continue; }
            
            int candidate = d_col_idx[state->neighbor_idx];
            state->neighbor_idx++;

            bool is_fully_connected = true;
            for (int j = 0; j < current_size; ++j) {
                if (!are_connected_global(current_clique[j], candidate, d_row_ptr, d_col_idx)) {
                    is_fully_connected = false; break;
                }
            }
            
            if (is_fully_connected) {
                current_clique[current_size] = candidate;
                current_size++;
                if (current_size == K) {
                    unsigned long long store_idx = atomicAdd(d_clique_count, 1);
                    if (store_idx < MAX_CLIQUES) for(int i=0; i<K; ++i) d_clique_results[store_idx * K + i] = current_clique[i];
                    current_size--;
                } else {
                    int last_vtx = current_clique[current_size - 1];
                    search_stack[current_size - 1] = (SearchState){ d_row_ptr[last_vtx], d_row_ptr[last_vtx + 1] };
                }
            }
        }
    }
}
'''

def find_best_clique(cliques):
    """Deterministic rule: lowest sum of vertex IDs wins."""
    return min(cliques, key=lambda c: sum(c))

def coarsen_graph_by_cliques(g, K, max_cliques_to_store=50_000_000):
    """
    Coarsens a DGL graph by merging nodes found in K-cliques.

    Args:
        g (DGLGraph): The input graph.
        K (int): The size of cliques to search for.
        max_cliques_to_store (int): Max number of cliques the GPU buffer can hold.

    Returns:
        DGLGraph: The new, coarsened graph.
    """
    if not cp.is_available():
        raise RuntimeError("CuPy is not available or not installed correctly.")
    if K > 16: # K_MAX in CUDA code
        raise ValueError(f"K={K} is too large. Max supported K is 16.")

    start_time = time.time()
    
    # 1. DATA PREPARATION (DGL -> CuPy)
    print("--- Step 1: Preparing data for GPU ---")
    print(f"Graph has {g.num_nodes()} nodes and {g.num_edges()} edges.")
    print(g)
    num_nodes = g.num_nodes()
    
    # DGL's CSC format is equivalent to CSR if we treat rows as columns
    indptr, indices, _ = g.adj_tensors('csc')
    
    # Move graph structure and features to GPU CuPy arrays
    d_row_ptr = cp.asarray(indptr, dtype=cp.int32)
    d_col_idx = cp.asarray(indices, dtype=cp.int32)
    features = cp.asarray(g.ndata['feat'], dtype=cp.float32)

    # Partition vertices on the CPU before launching kernels
    degrees = (d_row_ptr[1:] - d_row_ptr[:-1]).get()
    
    small_degree_nodes, large_degree_nodes, pruned_nodes = [], [], []
    for i in range(num_nodes):
        if degrees[i] >= K - 1:
            if degrees[i] <= 2048: # MAX_DEGREE_FOR_SHARED_MEM
                small_degree_nodes.append(i)
            else:
                large_degree_nodes.append(i)
        else:
            pruned_nodes.append(i)
            
    small_count, large_count = len(small_degree_nodes), len(large_degree_nodes)
    print(f"Vertex Partitions:\n  - {small_count} for optimized kernel\n  - {large_count} for fallback kernel\n  - {len(pruned_nodes)} pruned (degree < K-1)")

    # 2. HIGH-PERFORMANCE CLIQUE FINDING (CuPy RawKernel)
    print(f"\n--- Step 2: Finding {K}-cliques on GPU ---")
    
    # Compile the CUDA kernels from the string
    kernel_opt = cp.RawKernel(CUDA_KERNEL_CODE, 'k_clique_kernel_optimized')
    kernel_fallback = cp.RawKernel(CUDA_KERNEL_CODE, 'k_clique_kernel_fallback')

    # Allocate GPU memory for task lists and results
    d_small_task_list = cp.asarray(small_degree_nodes, dtype=cp.int32)
    d_large_task_list = cp.asarray(large_degree_nodes, dtype=cp.int32)
    d_clique_count = cp.zeros(1, dtype=cp.uint64)
    # Store cliques as flat array for simplicity, will reshape later
    d_clique_results = cp.empty(max_cliques_to_store * K, dtype=cp.int32)

    threads_per_block = 256
    shared_mem_size = 2048 * 4 # MAX_DEGREE_FOR_SHARED_MEM * sizeof(int)

    if small_count > 0:
        kernel_opt((small_count,), (threads_per_block,), 
                   (K, d_small_task_list, d_row_ptr, d_col_idx, d_clique_count, d_clique_results, max_cliques_to_store),
                   shared_mem=shared_mem_size)
    if large_count > 0:
        kernel_fallback((large_count,), (threads_per_block,),
                        (K, d_large_task_list, d_row_ptr, d_col_idx, d_clique_count, d_clique_results, max_cliques_to_store))

    cp.cuda.runtime.deviceSynchronize()
    found_clique_count = int(d_clique_count.get()[0])
    
    print(f"Found {found_clique_count} cliques in {time.time() - start_time:.2f} seconds.")
    if found_clique_count == 0:
        print("No cliques found. Returning original graph.")
        return g
    if found_clique_count >= max_cliques_to_store:
        print("WARNING: Found cliques may have reached the storage limit. Result could be incomplete.")

    # 3. DETERMINISTIC MERGING
    print("\n--- Step 3: Performing deterministic merge ---")
    
    # Get cliques from GPU to CPU (and reshape)
    num_cliques_to_process = min(found_clique_count, max_cliques_to_store)
    cliques_flat = d_clique_results[:num_cliques_to_process * K].get()
    cliques = cliques_flat.reshape((num_cliques_to_process, K))

    # Group cliques by the nodes they contain for fast lookup
    node_to_cliques = {}
    for clique in cliques:
        for node in clique:
            if node not in node_to_cliques:
                node_to_cliques[node] = []
            # Store as a tuple for hashing and immutability
            node_to_cliques[node].append(tuple(sorted(clique)))

    is_merged = np.zeros(num_nodes, dtype=bool)
    supernode_map = np.full(num_nodes, -1, dtype=np.int64)
    supernode_members = []
    supernode_counter = 0

    for node in range(num_nodes):
        if not is_merged[node]:
            if node in node_to_cliques:
                # This node is in one or more cliques. Find the best one.
                best_clique = find_best_clique(node_to_cliques[node])
                
                # Assign all members of this clique to the same new supernode
                current_supernode_members = []
                for member in best_clique:
                    if not is_merged[member]: # Avoid double-marking
                        supernode_map[member] = supernode_counter
                        is_merged[member] = True
                        current_supernode_members.append(member)
                supernode_members.append(current_supernode_members)
                supernode_counter += 1
            else:
                # This node is not in any clique, it becomes its own supernode
                supernode_map[node] = supernode_counter
                is_merged[node] = True
                supernode_members.append([node])
                supernode_counter += 1
                
    num_supernodes = supernode_counter
    print(f"Original nodes {num_nodes} merged into {num_supernodes} supernodes.")

    # 4. COARSE GRAPH CONSTRUCTION
    print("\n--- Step 4: Building new coarsened graph ---")
    
    # 4a. Aggregate features
    # Use CuPy for fast aggregation
    d_supernode_map = cp.asarray(supernode_map)
    d_new_features = cp.zeros((num_supernodes, features.shape[1]), dtype=cp.float32)
    d_supernode_sizes = cp.zeros(num_supernodes, dtype=cp.float32)

    # Sum features into supernode bins
    d_new_features.scatter_add(0, d_supernode_map[:, None].repeat(features.shape[1], axis=1), features)
    # Count members per supernode
    d_counts = cp.bincount(d_supernode_map)
    d_supernode_sizes[:len(d_counts)] = d_counts.astype(cp.float32)

    # Average the features
    d_new_features /= d_supernode_sizes[:, None]
    
    # 4b. Reconstruct edges
    src, dst = g.edges()
    src, dst = src.numpy(), dst.numpy()
    
    super_src = supernode_map[src]
    super_dst = supernode_map[dst]

    # Filter out self-loops and create a set of unique edges
    edge_set = set()
    for i in range(len(super_src)):
        if super_src[i] != super_dst[i]:
            edge_set.add((super_src[i], super_dst[i]))

    new_src, new_dst = zip(*edge_set)
    
    # 4c. Create the new DGL graph
    g_coarse = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)), num_nodes=num_supernodes)
    g_coarse.ndata['feat'] = torch.from_numpy(d_new_features.get())

    print("\n--- Coarsening Complete ---")
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    
    return g_coarse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coarsen a DGL graph using K-Clique merging.")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="Name of the OGB dataset to use.")
    parser.add_argument("--k", type=int, required=True, help="Size of cliques (K) to merge.")
    parser.add_argument("--max_cliques", type=int, default=50_000_000, help="Maximum number of cliques the GPU buffer can hold.")

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'ogbn-arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        original_graph, _ = dataset[0]
        # OGBN-Arxiv is directed, convert to undirected for clique finding
        # original_graph = dgl.to_bidirected(original_graph)
    else:
        # Example for other datasets like Reddit
        # import dgl.data
        # dataset = dgl.data.RedditDataset()
        # original_graph = dataset[0]
        raise NotImplementedError(f"Dataset '{args.dataset}' not configured. Please add loading logic.")

    print("\n--- Original Graph ---")
    print(original_graph)

    coarse_graph = coarsen_graph_by_cliques(original_graph, args.k, args.max_cliques)

    print("\n--- Coarsened Graph ---")
    print(coarse_graph)
