# drazin.py
"""Volume 1: The Drazin Inverse.
Aidan Wittman
MTH420 Prof. Gibson Spring 2022
6/7/2022
"""

import numpy as np
from scipy import linalg as la


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    if(np.allclose(np.matmul(A,Ad), np.matmul(Ad,A))) is True:
        a = 0
    else:
        return False
    if(np.allclose(np.matmul(np.linalg.matrix_power(A, k+1),Ad), np.linalg.matrix_power(A,k))) is True:
        a = 0
    else:
        return False
    if(np.allclose(np.matmul(np.matmul(Ad,A),Ad), Ad)) is True:
        a = 0
    else:
        return False
    return True

    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    nrows, ncols = np.shape(A)
    f = lambda x: abs(x) > tol
    g = lambda x: abs(x) <= tol
    T1, Q1, k1 = la.schur(A, sort=f)
    T2, Q2, k2 = la.schur(A, sort=g)
    U1 = Q1[:,:k1]
    U2 = Q2[:,:nrows - k1]
    U = np.hstack((U1,U2))
    Uinv = la.inv(U)
    V = np.matmul(np.matmul(Uinv,A),U)
    Z = np.zeros((nrows,nrows))
    if k1 != 0:
        VFX = V[:k1,:k1]
        Minv = la.inv(VFX)
        Z[:k1,:k1] = Minv
    R = np.matmul(np.matmul(U,Z),Uinv)
    return(R)

    raise NotImplementedError("Problem 2 Incomplete")

def laplacian(A):
    """Compute the Laplacian matrix of the adjacency matrix A,
    as well as the second smallest eigenvalue.
    
    Parameters:
        A ((n,n) ndarray) adjacency matrix for an undirected weighted graph.
        
    Returns:
        L ((n,n) ndarray): the Laplacian matrix of A
    """
    D = A.sum(axis=1) #The degree of each vertex (either axis).
    return np.diag(D) - A

# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    nrows, ncols = np.shape(A)
    L = laplacian(A)
    R = np.empty((nrows,ncols))
    for i in range(0,nrows):
        for j in range(0,nrows):
            Lfx = L.copy()
            Lfx[j] = np.eye(nrows)[j]
            Ld = drazin_inverse(Lfx)
            if i == j:
                R[i,j] = 0
            else:
                R[i,j] = Ld[i,i]
    return(R)


    raise NotImplementedError("Problem 3 Incomplete")


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        raise NotImplementedError("Problem 4 Incomplete")


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")

def main():
    A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
    B = np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
    AD = np.array([[1,-3,9,81],[0,1,-3,-18],[0,0,1,3],[0,0,0,0]])
    BD = np.zeros((3,3))
    Ares = is_drazin(A,AD,1)
    Bres = is_drazin(B,BD,3)
    print('Is AD a Drazin Inverse of A?:', Ares,'\nIs BD a Drazin Inverse of B?:', Bres)
    R = drazin_inverse(A)

    C = np.array([[9,3,0,0],[1,2,3,0],[3,4,5,6],[0,0,0,0]])
    ck = index(C)
    print(ck)
    S = drazin_inverse(C)
    Cres = is_drazin(C, S, ck)
    print(Cres)

    E = np.array([[0,1,0,0,1,1],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[1,0,0,1,0,0]])
    Eres = effective_resistance(E)
    print(Eres)

    F = np.array([[0,3,0,1,0,0],[3,0,0,0,0,0],[0,0,0,1,0,0],[1,0,1,0,2,0.5],[0,0,0,2,0,1],[0,0,0,0.5,1,0]]) #Replacing (1,4) and (4,1) with 1's instead of 0's yields a valid matrix, whereas having them be 0's yields an error. Why?
    Fres = effective_resistance(F)
    print(Fres)

main()
