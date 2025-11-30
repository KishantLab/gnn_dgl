"""
## 1. Fixing the Compilation Error (error: expected an expression)
You are right, the code I provided has a syntax error. The issue is in how a SearchState object is created inside the kernel.

The C-style syntax search_stack[...] = (SearchState){ value1, value2 }; is called a compound literal. While some compilers accept this in C++, the NVIDIA NVRTC compiler used by CuPy is stricter and does not support it.

The Fix: We need to use standard C++ initialization. We'll create a temporary SearchState variable and then assign it.

Old (Incorrect) Code:
search_stack[current_size - 1] = (SearchState){ d_row_ptr[v], d_row_ptr[v + 1] };

New (Correct) Code:

C++

SearchState newState;
newState.neighbor_idx = d_row_ptr[v];
newState.neighbor_end_idx = d_row_ptr[v + 1];
search_stack[current_size - 1] = newState;
I will apply this fix to the CUDA kernel string in the final code.

## 2. Integrating Your Graph Conversion Kernels
Your Find_size and Convert kernels are the perfect solution for the required pre-processing step. My previous script was missing this and incorrectly assumed the input graph was already in the special directed format.

The new, correct workflow will be:

Load the original undirected graph from DGL.

Run your Find_size kernel to calculate the out-degree of each node for the new directed graph.

Perform a cumsum (prefix-sum) on the GPU to calculate the offsets for the new row_ptr array.

Run your Convert kernel to build the final degree-ordered, directed graph on the GPU.

Run our k-clique kernels on this newly created directed graph.

Proceed with the merging and coarsening steps as before.

## 3. The Complete, Corrected, and Integrated Code
This final script incorporates the bug fix and fully integrates your graph conversion logic for a complete, end-to-end workflow.
"""

import dgl
import torch
import cupy as cp
import numpy as np
import argparse
import time
from ogb.nodeproppred import DglNodePropPredDataset

# =========================================================================
# KERNEL 1: Your Undirected -> Directed Graph Conversion Kernels
# =========================================================================
UNDIR_TO_DIR_KERNELS = r'''
// Note: Using int for compatibility with DGL/CuPy defaults instead of unsigned long long
extern "C" __global__
void Find_size(const int *d_row_ptr, const int *d_col_idx, int *temp_arr, int row_ptr_s)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < row_ptr_s)
    {
        int count = 0;
        int deg_src = d_row_ptr[id+1] - d_row_ptr[id];
        for(int j = d_row_ptr[id]; j < d_row_ptr[id+1]; j++)
        {
            int dst_node = d_col_idx[j];
            int deg_dst = d_row_ptr[dst_node+1] - d_row_ptr[dst_node];
            if(deg_src < deg_dst)
            {
                count++;
            }
            else if(deg_src == deg_dst)
            {
                // Standard tie-breaking: edge goes from lower ID to higher ID
                if(dst_node > id)
                {
                    count++;
                }
            }
        }
        temp_arr[id] = count;
    }
}

extern "C" __global__
void Convert(const int *d_row_ptr, const int *d_col_idx, const int *temp_arr_sum, int *d_row_ptr_Dir, int *d_col_idx_Dir, int row_ptr_s)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < row_ptr_s)
    {
        int pos = (id == 0) ? 0 : temp_arr_sum[id-1];
        int deg_src = d_row_ptr[id+1] - d_row_ptr[id];
        for(int j = d_row_ptr[id]; j < d_row_ptr[id+1]; j++)
        {
            int dst_node = d_col_idx[j];
            int deg_dst = d_row_ptr[dst_node+1] - d_row_ptr[dst_node];
            if(deg_src < deg_dst)
            {
                d_col_idx_Dir[pos] = dst_node;
                pos++;
            }
            else if(deg_src == deg_dst)
            {
                if(dst_node > id)
                {
                    d_col_idx_Dir[pos] = dst_node;
                    pos++;
                }
            }
        }
        d_row_ptr_Dir[id+1] = pos;
    }
    if (id == 0) d_row_ptr_Dir[0] = 0;
}
'''

# =========================================================================
# KERNEL 2: Our K-Clique Finding Kernels (with the compilation fix)
# =========================================================================
K_CLIQUE_KERNELS = r'''
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
        
        SearchState newState;
        newState.neighbor_idx = d_row_ptr[v];
        newState.neighbor_end_idx = d_row_ptr[v + 1];
        search_stack[current_size - 1] = newState;

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
                    SearchState nextState;
                    nextState.neighbor_idx = d_row_ptr[last_vtx];
                    nextState.neighbor_end_idx = d_row_ptr[last_vtx + 1];
                    search_stack[current_size - 1] = nextState;
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

        SearchState newState;
        newState.neighbor_idx = d_row_ptr[v];
        newState.neighbor_end_idx = d_row_ptr[v + 1];
        search_stack[current_size - 1] = newState;

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
                    SearchState nextState;
                    nextState.neighbor_idx = d_row_ptr[last_vtx];
                    nextState.neighbor_end_idx = d_row_ptr[last_vtx + 1];
                    search_stack[current_size - 1] = nextState;
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
    if not cp.is_available():
        raise RuntimeError("CuPy is not available or not installed correctly.")
    if K > 16:
        raise ValueError(f"K={K} is too large. Max supported K is 16.")

    start_time = time.time()
    num_nodes = g.num_nodes()
    
    # 1. CONVERT UNDIRECTED TO DIRECTED ON GPU
    print("--- Step 1: Converting graph to degree-ordered directed graph on GPU ---")
    
    # Ensure graph is undirected first, and critically, copy node data
    g_undir = dgl.to_bidirected(g, copy_ndata=True)
    undir_indptr, undir_indices, _ = g_undir.adj_tensors('csc')
    
    d_undir_row_ptr = cp.asarray(undir_indptr, dtype=cp.int32)
    d_undir_col_idx = cp.asarray(undir_indices, dtype=cp.int32)

    conversion_module = cp.RawModule(code=UNDIR_TO_DIR_KERNELS)
    find_size_kernel = conversion_module.get_function('Find_size')
    convert_kernel = conversion_module.get_function('Convert')

    block_size = 1024
    grid_size = (num_nodes + block_size - 1) // block_size
    
    d_temp_arr = cp.empty(num_nodes, dtype=cp.int32)
    find_size_kernel((grid_size,), (block_size,), (d_undir_row_ptr, d_undir_col_idx, d_temp_arr, num_nodes))
    
    d_temp_arr_sum = cp.cumsum(d_temp_arr)
    new_num_edges = int(d_temp_arr_sum[-1].get()) if num_nodes > 0 and d_temp_arr_sum.size > 0 else 0

    d_dir_row_ptr = cp.empty(num_nodes + 1, dtype=cp.int32)
    d_dir_col_idx = cp.empty(new_num_edges, dtype=cp.int32)
    
    convert_kernel((grid_size,), (block_size,), (d_undir_row_ptr, d_undir_col_idx, d_temp_arr_sum, d_dir_row_ptr, d_dir_col_idx, num_nodes))
    
    print(f"Conversion complete. New directed graph has {new_num_edges} edges.")

    # 2. K-CLIQUE FINDING on the NEW Directed Graph
    print(f"\n--- Step 2: Finding {K}-cliques on the new directed graph ---")
    
    features = cp.asarray(g.ndata['feat'], dtype=cp.float32)
    degrees = (d_dir_row_ptr[1:] - d_dir_row_ptr[:-1]).get()
    
    small_degree_nodes, large_degree_nodes, pruned_nodes = [], [], []
    for i in range(num_nodes):
        if degrees[i] >= K - 1:
            if degrees[i] <= 2048:
                small_degree_nodes.append(i)
            else:
                large_degree_nodes.append(i)
        else:
            pruned_nodes.append(i)
            
    small_count, large_count = len(small_degree_nodes), len(large_degree_nodes)
    print(f"Vertex Partitions:\n  - {small_count} for optimized kernel\n  - {large_count} for fallback kernel\n  - {len(pruned_nodes)} pruned (out-degree < K-1)")

    clique_module = cp.RawModule(code=K_CLIQUE_KERNELS)
    kernel_opt = clique_module.get_function('k_clique_kernel_optimized')
    kernel_fallback = clique_module.get_function('k_clique_kernel_fallback')

    d_small_task_list = cp.asarray(small_degree_nodes, dtype=cp.int32)
    d_large_task_list = cp.asarray(large_degree_nodes, dtype=cp.int32)
    d_clique_count = cp.zeros(1, dtype=cp.uint64)
    d_clique_results = cp.empty(max_cliques_to_store * K, dtype=cp.int32)

    threads_per_block = 256
    shared_mem_size = 2048 * 4

    if small_count > 0:
        kernel_opt((small_count,), (threads_per_block,), 
                   (K, d_small_task_list, d_dir_row_ptr, d_dir_col_idx, d_clique_count, d_clique_results, max_cliques_to_store),
                   shared_mem=shared_mem_size)
    if large_count > 0:
        kernel_fallback((large_count,), (threads_per_block,),
                        (K, d_large_task_list, d_dir_row_ptr, d_dir_col_idx, d_clique_count, d_clique_results, max_cliques_to_store))

    cp.cuda.runtime.deviceSynchronize()
    found_clique_count = int(d_clique_count.get()[0])
    
    print(f"Found {found_clique_count} cliques in {time.time() - start_time:.2f} seconds.")
    if found_clique_count == 0:
        print("No cliques found to merge. Returning original graph.")
        return g

    # 3. DETERMINISTIC MERGING
    print("\n--- Step 3: Performing deterministic merge ---")
    
    num_cliques_to_process = min(found_clique_count, max_cliques_to_store)
    cliques_flat = d_clique_results[:num_cliques_to_process * K].get()
    cliques = cliques_flat.reshape((num_cliques_to_process, K))

    node_to_cliques = {}
    for clique in cliques:
        clique_tuple = tuple(sorted(clique))
        for node in clique_tuple:
            if node not in node_to_cliques:
                node_to_cliques[node] = set()
            node_to_cliques[node].add(clique_tuple)

    is_merged = np.zeros(num_nodes, dtype=bool)
    supernode_map = np.full(num_nodes, -1, dtype=np.int64)
    supernode_counter = 0

    for node in range(num_nodes):
        if not is_merged[node]:
            if node in node_to_cliques:
                best_clique = find_best_clique(list(node_to_cliques[node]))
                for member in best_clique:
                    if not is_merged[member]:
                        supernode_map[member] = supernode_counter
                        is_merged[member] = True
                supernode_counter += 1
            else:
                supernode_map[node] = supernode_counter
                is_merged[node] = True
                supernode_counter += 1
                
    num_supernodes = supernode_counter
    print(f"Original nodes {num_nodes} merged into {num_supernodes} supernodes.")

    # 4. COARSE GRAPH CONSTRUCTION
    print("\n--- Step 4: Building new coarsened graph ---")
    
    d_supernode_map = cp.asarray(supernode_map)
    d_new_features = cp.zeros((num_supernodes, features.shape[1]), dtype=cp.float32)
    
    d_new_features.scatter_add(0, d_supernode_map[:, None].repeat(features.shape[1], axis=1), features)
    d_counts = cp.bincount(d_supernode_map)
    d_supernode_sizes = cp.zeros(num_supernodes, dtype=cp.float32)
    d_supernode_sizes[:len(d_counts)] = d_counts.astype(cp.float32)
    
    # Avoid division by zero for supernodes that might be empty (should not happen in this logic)
    d_supernode_sizes[d_supernode_sizes == 0] = 1
    d_new_features /= d_supernode_sizes[:, None]
    
    src, dst = g_undir.edges()
    src, dst = src.numpy(), dst.numpy()
    
    super_src = supernode_map[src]
    super_dst = supernode_map[dst]

    edge_set = set()
    for i in range(len(super_src)):
        if super_src[i] != super_dst[i]:
            # Ensure consistent edge direction (e.g., smaller to larger) to make it undirected
            u, v = sorted((super_src[i], super_dst[i]))
            edge_set.add((u, v))

    if not edge_set:
        print("Warning: Coarsened graph has no edges.")
        g_coarse = dgl.graph(([],[]), num_nodes=num_supernodes)
    else:
        new_src, new_dst = zip(*edge_set)
        g_coarse = dgl.graph((torch.tensor(new_src), torch.tensor(new_dst)), num_nodes=num_supernodes)
        g_coarse = dgl.to_bidirected(g_coarse) # Make it undirected again

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
        # This graph is directed, but clique finding should happen on its underlying undirected structure.
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' not configured. Please add loading logic.")

    print("\n--- Original Graph ---")
    print(original_graph)

    coarse_graph = coarsen_graph_by_cliques(original_graph, args.k, args.max_cliques)

    print("\n--- Coarsened Graph ---")
    print(coarse_graph)
