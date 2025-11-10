import numpy as np

def get_adjancency_matrix(filePath):
    """
    Cunstruct adjancency matrix from data
    input: file path
    output: id_to_url = dict of id and url
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

def pageRank_simple(id_to_url, adj_matrix):
    """
    Rank websites simple, problem with subwebs
    input: id_to_url = dict of id and url
            adj_matrix = adjancency matrix
    output: result = dict {id:[score, url]}
    """
    num_pages = adj_matrix.shape[1]

    link_matrix = np.zeros_like(adj_matrix)

    # vector of outgoing links from each page
    outgoing = adj_matrix.sum(axis = 0)

    # build link matrix from adjancency matrix
    for src in range(num_pages):
        if outgoing[src] == 0:
            continue
        for dst in range(num_pages):
            if adj_matrix[src, dst] == 1:
                # devide the pages vote based on the number of outgoing links
                link_matrix[src, dst] = 1.0/outgoing[dst]

    x = np.ones(num_pages)/num_pages
    max_iters = 10000
    tol = 0.00000001

    eig_vector = power_method(link_matrix, x, max_iters, tol)

    result = {}
    for page_id, url in id_to_url.items():
        result[page_id] = [eig_vector[page_id], url]

    return result

def power_method(A, x, max_iters, tol):
    """
    Compute eigen vector of matrix using power method
    input:  A = matrix
            x = initializing vector
            max_iters = maximum amount of iterations
            tol = tolerance

    output: x = eigen vector of input matrix
    """

    for _ in range(max_iters):
        x_next = np.dot(A, x)

        # normalise in the L1 norm
        x_next /= x_next.sum()

        if np.linalg.norm(x_next - x, 1) < tol:
            break

        x = x_next

    return x

def pageRank(id_to_url, adj_matrix):
    """
    Rank websites, deals with subwebs
    input: id_to_url = dict of id and url
            adj_matrix = adjancency matrix
    output: result = dict {id:[score, url]}
    """
    num_pages = adj_matrix.shape[1]

    link_matrix = np.zeros_like(adj_matrix)

    outgoing = adj_matrix.sum(axis = 0)

    # build link matrix from adjancency matrix
    for src in range(num_pages):
        if outgoing[src] == 0:
            continue
        for dst in range(num_pages):
            if adj_matrix[src, dst] == 1:
                # devide the pages vote based on the number of outgoing links
                link_matrix[src, dst] = 1.0/outgoing[dst]

    # modify the link matrix
    S = np.ones_like(adj_matrix)/num_pages

    m = 0.15

    # modified link matrix
    M = (1 - m) * link_matrix + m * S

    x = np.ones(num_pages)/num_pages
    max_iters = 10000
    tol = 0.00000001

    eigen_vector = power_method(M, x, max_iters, tol)

    result = {}

    for page_id, url in id_to_url.items():
        result[page_id] = [eigen_vector[page_id - 1], url]

    return result
    
def display_results(result_dict):
    """
    Display of the ranked websites
    input: dict of results
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