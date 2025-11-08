import os
import functions as fn
import numpy as np

# filePath = os.path.join(os.getcwd(), "hollins.dat")

# id_to_url, A = fn.get_adjancency_matrix(filePath=filePath)

# print(f"the dictioanry looks like: {id_to_url}")
# print(f"the matrix has: {np.count_nonzero(A)} nonzero entries")

# test simple pageRank
test = np.array([
    [0, 0, 1, 1],  
    [1, 0, 0, 0],  
    [1, 1, 0, 1],  
    [1, 1, 0, 0]   
], dtype=float)

id_to_url = {
    0: "http://site1.com",
    1: "http://site2.com",
    2: "http://site3.com",
    3: "http://site4.com"
}

ranking = fn.pageRank_simple(id_to_url, test)

ranking_medium = fn.pageRank_medium(id_to_url, test)

# test subweb
test_subweb = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0]
], dtype=float)

id_to_url_subweb = {
    0: "http://site1.com",
    1: "http://site2.com",
    2: "http://site3.com",
    3: "http://site4.com",
    4: "http://site5.com"
}

ranking_subwebs = fn.pageRank_medium(id_to_url_subweb, test_subweb)

print("=== Computed PageRank results ===")
print(ranking) 

print("=== Computed medium PageRank results ===")
print(ranking_medium)

print("=== Computed subweb PageRank results ===")
print(ranking_subwebs)