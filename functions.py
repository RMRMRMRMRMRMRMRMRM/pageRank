import numpy as np
import sparse as sp
from collections import defaultdict
import time
from scipy.sparse import csr_matrix

def get_adjancency_matrix(filePath):
    """
    Cunstruct adjancency matrix from data.

    Input:  file path

    Output: id_to_url = dict of id and url
            A = adjancency matrix
    """
    with open(file = filePath, mode = 'r', encoding='utf-8', errors='ignore') as f:
        num_websites, num_connections = map(int, f.readline().split())

        id_to_url = {}

        # build id_to_url dict
        for _ in range(num_websites):
            line = f.readline().strip()
            idx, url = line.split(maxsplit = 1)
            id_to_url[int(idx)] = url

        # build adjancency matrix from data
        A = np.zeros((num_websites, num_websites))

        # fill adjancency matrix from data
        for line in f:
            src, dst = map(int, line.split())
            A[dst - 1, src - 1] = 1.0

        return id_to_url, A


def power_method(A, x, max_iters, tol, m):
    """
    Compute eigen vector of matrix using power method.

    Input:  A = matrix
            x = initializing vector
            max_iters = maximum amount of iterations
            tol = tolerance
            m = weight

    Output: x = eigen vector of input matrix
    """
    s = np.ones_like(x, dtype=float)
    for _ in range(max_iters):
        x_next = (1 - m) * A @ x + (m / A.shape[0]) * s

        # normalise in the L1 norm
        x_next /= x_next.sum()

        if np.linalg.norm(x_next - x, 1) < tol:
            break

        x = x_next

    return x

def pageRank(id_to_url, adj_matrix):
    """
    Rank websites, deals with subwebs.

    Input: id_to_url = dict of id and url
            adj_matrix = adjancency matrix

    Output: result = dict {id:[score, url]}
    """
    num_pages = adj_matrix.shape[1]

    link_matrix = np.zeros_like(adj_matrix)

    outgoing = adj_matrix.sum(axis = 0)

    # build link matrix from adjancency matrix
    for src in range(num_pages):
        for dst in range(num_pages):
            if adj_matrix[src, dst] == 1:
                if outgoing[dst] == 0:
                    continue

                # devide the pages vote based on the number of outgoing links
                link_matrix[src, dst] = 1.0/outgoing[dst]


    x = np.ones(num_pages)/num_pages
    max_iters = 10000
    tol = 0.00000001
    m = 0.15

    # compute eigen vector and log time 
    start = time.perf_counter()
    eigen_vector = power_method(link_matrix, x, max_iters, tol, m)
    end = time.perf_counter()

    print("Dense matrix log info")
    print(f"Runtime dense matrix: {end - start:.6f} seconds")
    print(f"Memory taken up by the dense matrix: {link_matrix.nbytes} bytes")
    print(140*"-")

    result = {}

    for page_id, url in id_to_url.items():
        result[page_id] = [eigen_vector[page_id - 1], url]

    return result

def pageRank_scipy(id_to_url, adj_matrix):
    """
    Rank websites, deals with subwebs.

    Input: id_to_url = dict of id and url
            adj_matrix = adjancency matrix

    Output: result = dict {id:[score, url]}
    """
    num_pages = adj_matrix.shape[1]

    link_matrix = np.zeros_like(adj_matrix)

    outgoing = adj_matrix.sum(axis = 0)

    # build link matrix from adjancency matrix
    for src in range(num_pages):
        for dst in range(num_pages):
            if adj_matrix[src, dst] == 1:
                if outgoing[dst] == 0:
                    continue

                # devide the pages vote based on the number of outgoing links
                link_matrix[src, dst] = 1.0/outgoing[dst]


    x = np.ones(num_pages)/num_pages
    max_iters = 10000
    tol = 0.00000001
    m = 0.15

    link_sparse = csr_matrix(link_matrix)

    mem_sparse = link_sparse.data.nbytes + link_sparse.indices.nbytes + link_sparse.indptr.nbytes

    # compute eigen vector and log time 
    start = time.perf_counter()
    eigen_vector = power_method(link_sparse, x, max_iters, tol, m)
    end = time.perf_counter()

    print("Scipy sparse matrix log info")
    print(f"Runtime scipy sparse matrix: {end - start:.6f} seconds")
    print(f"Memory taken up by the scipy sparse matrix: {mem_sparse} bytes")
    print(140*"-")

    result = {}

    for page_id, url in id_to_url.items():
        result[page_id] = [eigen_vector[page_id - 1], url]

    return result

def display_results(result_dict):
    """
    Display of the ranked websites.

    Input:  dict of results

    Output: none
    """
    ordered_dict = sorted(result_dict.items(), key=lambda item: item[1][0], reverse=True)
    
    n = min(5, len(ordered_dict))
    top_n = ordered_dict[:n]
    bottom_n = ordered_dict[-n:]

    print("=== Top pages ===")
    print(140*"-")
    for i in range(n):
        print(f"ID:{top_n[i][0]:<5} | url: {top_n[i][1][1]:<90} | score: {top_n[i][1][0]:<.7f}")
    print(140*"-")

    print("=== Bottom pages ===")
    print(140*"-")
    for i in range(n):
        print(f"ID:{bottom_n[-i - 1][0]:<5} | url: {bottom_n[-i - 1][1][1]:<90} | score: {bottom_n[-i - 1][1][0]:<.7f}")
    print(140*"-")  

    return


def get_adjancency_matrix_sparse(filePath):
    """
    Cunstruct sparse adjancency matrix from data.

    Input:  file path

    Output: id_to_url = dict of id and url
            A = adjancency matrix
    """
    with open(file = filePath, mode = 'r', encoding='utf-8', errors='ignore') as f:
        num_websites, num_connections = map(int, f.readline().split())

        id_to_url = {}

        # build id_to_url dict
        for _ in range(num_websites):
            line = f.readline().strip()
            idx, url = line.split(maxsplit = 1)
            id_to_url[int(idx)] = url

        # build adjancency matrix from data
        links = defaultdict(set)

        for line in f:
            src, dst = map(int, line.split())
            links[dst].add(src)

        AA = np.ones(num_connections)
        JA = []
        IA = [1]

        # using 1 based indexing
        for row in range(1, num_websites + 1):
            filled_collumns = sorted(links[row])
            
            JA.extend(filled_collumns)
            IA.append(len(filled_collumns) + IA[row - 1])

        # convert to np array
        JA = np.array(JA, dtype=np.int64)
        IA = np.array(IA, dtype=np.int64)

        # fill sparse adjancency matrix from data
        A = sp.SparseMatrixCSR(AA, JA, IA, [num_websites, num_websites])

        return id_to_url, A

def pageRank_sparse(id_to_url, adj_matrix):
    """
    Rank websites, deals with subwebs.

    Input:  id_to_url = dict of id and url
            adj_matrix = adjancency matrix

    Output: result = dict {id:[score, url]}
    """
    num_pages = adj_matrix.shape[1]

    link_data = np.zeros_like(adj_matrix.data)

    outgoing = np.zeros(num_pages)

    # sum the outgoing links for each page
    # -1 is used because of the 1 based indexing
    for i in adj_matrix.indices:
        outgoing[i - 1] += 1

    # build the link matrix data, everything else is same as adj_matrix
    for idx, col in enumerate(adj_matrix.indices):
        if outgoing[col - 1] == 0:
            link_data[idx] = 0.0
        else:
            link_data[idx] = 1.0/outgoing[col - 1]

    # initialize link matrix the same as adj matrix
    link_matrix = sp.SparseMatrixCSR(link_data, adj_matrix.indices, adj_matrix.indptr, adj_matrix.shape)

    m = 0.15

    x = np.ones(num_pages)/num_pages
    max_iters = 10000
    tol = 0.00000001

    # compute eigen vector and log time
    start = time.perf_counter()
    eigen_vector = power_method_sparse(link_matrix, x, max_iters, tol, m)
    end = time.perf_counter()

    mem_sparse = link_matrix.data.nbytes + link_matrix.indices.nbytes + link_matrix.indptr.nbytes
    
    print("Sparse matrix log info")
    print(f"Runtime sparse matrix: {end - start:.6f} seconds")
    print(f"Memory taken up by the sparse matrix: {mem_sparse} bytes")
    print(140*"-")

    result = {}

    for page_id, url in id_to_url.items():
        result[page_id] = [eigen_vector[page_id - 1], url]

    return result
    
def power_method_sparse(A, x, max_iters, tol, m):
    """
    Compute eigen vector of matrix using power method.

    Input:  A = SparceMatrixCSR matrix
            x = initializing vector
            max_iters = maximum amount of iterations
            tol = tolerance
            m = weight

    Output: x = eigen vector of input matrix
    """
    s = np.ones_like(x, dtype=float)
    for _ in range(max_iters):
        x_next = (1 - m) * A.matVec(x) + (m / A.shape[0]) * s

        # normalise in the L1 norm
        x_next /= x_next.sum()

        if np.linalg.norm(x_next - x, 1) < tol:
            break

        x = x_next

    return x


