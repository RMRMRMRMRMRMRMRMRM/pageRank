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