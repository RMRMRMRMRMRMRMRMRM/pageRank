import os
import functions as fn
import numpy as np

filePath = os.path.join(os.getcwd(), "hollins.dat")

id_to_url, A = fn.get_adjancency_matrix(filePath=filePath)

print(f"the dictioanry looks like: {id_to_url}")
print(f"the matrix has: {np.count_nonzero(A)} nonzero entries")
