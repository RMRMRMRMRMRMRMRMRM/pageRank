import os
import functions as fn
import numpy as np

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

# test with weighted pagerank
ranking_weighted = fn.pageRank(id_to_url, test)

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

ranking_subwebs = fn.pageRank(id_to_url_subweb, test_subweb)

# compute real data
filePath = os.path.join(os.getcwd(), "hollins.dat")

id_to_url, A = fn.get_adjancency_matrix(filePath=filePath)

ranking_main = fn.pageRank(id_to_url, A)


# display results
print("=== Computed non weighted PageRank results ===")
fn.display_results(ranking)

print("=== Computed weighted PageRank results ===")
fn.display_results(ranking_weighted)

print("=== Computed weighted PageRank results for a subweb ===")
fn.display_results(ranking_subwebs)

print("=== Computed weighted PageRank results for given data ===")
fn.display_results(ranking_main)