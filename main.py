import os
import functions as fn
import numpy as np

# get filepath
filePath = os.path.join(os.getcwd(), "hollins.dat")

# get real data and compute ranking
id_to_url, A = fn.get_adjancency_matrix(filePath=filePath)
id_to_url_sparse, A_sparse = fn.get_adjancency_matrix_sparse(filePath)

ranking_dense = fn.pageRank(id_to_url, A)
ranking_sparse = fn.pageRank_sparse(id_to_url_sparse, A_sparse)
ranking_scipy_sparse = fn.pageRank_scipy(id_to_url, A)

# display results
print("=== Computed PageRank results for given data using dense matrix ===")
fn.display_results(ranking_dense)

print("=== Computed PageRank results for given data using self implemented CSR matrix ===")
fn.display_results(ranking_sparse)

print("=== Computed PageRank results for given data using scipy CSR matrix ===")
fn.display_results(ranking_scipy_sparse)
