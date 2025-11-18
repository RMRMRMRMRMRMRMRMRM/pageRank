import numpy as np

class SparseMatrixCSR:
    '''
    Representation of matrix in CSR format.
    
    Attributes
    ----------
    data : np.ndarray
        Nonzero values of the matrix.
    indices : np.ndarray
        Column indices corresponding to the values in `data`.
    indptr : np.ndarray
        Row pointer array, expected to be 1 based indexing.
    shape : tuple[int, int]
        Shape of the matrix (rows, columns).

    Methods
    -------
    matvec(x)
        Perform matrix–vector multiplication.
    '''
    def __init__(self, data, indices, indptr, shape):
        self.data = np.array(data, dtype=float)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.shape = shape

    def matVec(self, x):
        '''
        Multiplies self @ x

        Input: vector x
        Output: result = self @ x

        Warning
        ----------
        Expected 1 based indexing.
        '''
        n_rows = self.shape[0]
        result = np.zeros(self.shape[0], dtype=float)
        
        for i in range(n_rows):
            # here I use -1 because of the 1 based indexing for the method
            start = self.indptr[i] - 1
            end = self.indptr[i + 1] - 1

            if start == end:
                continue

            cols = self.indices[start:end] -1
            row = self.data[start:end]

            result[i] = np.dot(row, x[cols])

            # unoptimised version
            # for indx in range(start, end):
            #     result[i] +=  self.data[indx] * x[self.indices[indx] - 1]

        return result