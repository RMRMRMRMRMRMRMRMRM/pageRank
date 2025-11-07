import numpy as np

def get_adjancency_matrix(filePath):

    with open(file = filePath, mode = 'r', encoding='utf-8', errors='ignore') as f:
        num_websites, num_connections = map(int, f.readline().split())

        id_to_url = {}
        
        for _ in range(num_websites):
            line = f.readline().strip()
            idx, url = line.split(maxsplit = 1)
            id_to_url[int(idx)] = url

        A = np.zeros((num_websites, num_websites))

        for line in f:
            idx, idy = map(int, line.split())
            A[idx - 1, idy - 1] = 1.0   
        
        return id_to_url, A

def pageRank_simple(id_to_url, adj_matrix):
    
    num_pages = adj_matrix.shape[1]

    link_matrix = np.zeros_like(adj_matrix)

    outgoing = adj_matrix.sum(axis = 0)

    for src in range(num_pages):
        if outgoing[src] == 0:
            continue
        for dst in range(num_pages):
            if adj_matrix[src, dst] == 1:
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
    
    for _ in range(max_iters):
        
        x_next = np.dot(A, x)
        x_next /= x_next.sum()

        if np.linalg.norm(x_next - x, 1) < tol:
            break
        
        x = x_next

    return x

