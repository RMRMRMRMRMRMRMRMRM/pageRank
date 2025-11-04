import os
import functions as fn
import numpy as np

# filePath = os.path.join(os.getcwd(), "hollins.dat")

# id_to_url, A = fn.get_adjancency_matrix(filePath=filePath)

# print(f"the dictioanry looks like: {id_to_url}")
# print(f"the matrix has: {np.count_nonzero(A)} nonzero entries")

# test simple pageRank
test = np.array([
    [0, 1, 1, 1],  
    [0, 0, 1, 1],  
    [1, 0, 0, 0],  
    [1, 0, 1, 0]   
], dtype=float)

id_to_url = {
    0: "http://site1.com",
    1: "http://site2.com",
    2: "http://site3.com",
    3: "http://site4.com"
}

ranking = fn.pageRank_simple(id_to_url, test)

print("=== Computed PageRank results ===")
print(ranking) 