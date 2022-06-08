# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Aidan Wittman
MTH420 Prof. Gibson Spring 2022
6/7/2022
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q, R = la.qr(A, mode="economic")
    QT = np.transpose(Q)
    z = np.matmul(QT,b)
    x = la.solve_triangular(R,z)
    return(x)
    raise NotImplementedError("Problem 1 Incomplete")

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    housing = np.load("/Users/aidan/OneDrive/Desktop/.vscode/housing.npy") #Change the path to whatever housing.npy becomes: it only works for me with this type of path. /Users/<name>/ etc.
    print(housing)
    x = housing[:,0]
    y = housing[:,1]
    A = np.empty((33,2))
    B = np.empty((33,1))
    for i in range(0,33):
        A[i,0] = x[i]
        A[i,1] = 1
        B[i,0] = y[i]
    z = least_squares(A,B)
    print(z)
    a = z[0,0]
    b = z[1,0]
    plt.scatter(x,y)
    plt.plot(x, a*x + b)
    plt.show()
    return()
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    housing = np.load("/Users/aidan/OneDrive/Desktop/.vscode/housing.npy") #Change the path to whatever housing.npy becomes: it only works for me with this type of path. /Users/<name>/ etc.
    xh = housing[:,0]
    yh = housing[:,1]
    B = np.empty((33,1))
    for i in range(0,33):
        B[i] = yh[i]
    x = np.linspace(0,16,100)
    three = np.vander(xh,3)
    six = np.vander(xh,6)
    nine = np.vander(xh,9)
    twelve = np.vander(xh,12)
    tx = least_squares(three, B)
    sx = least_squares(six, B)
    nx = least_squares(nine, B)
    twx = least_squares(twelve, B)
    txf = np.ravel(tx)
    tf = np.poly1d(txf)
    sxf = np.ravel(sx)
    sf = np.poly1d(sxf)
    nxf = np.ravel(nx)
    nf = np.poly1d(nxf)
    twxf = np.ravel(twx)
    twf = np.poly1d(twxf)
    ax1 = plt.subplot(2,2,1)
    ax1.plot(x, tf(x), "r-")
    ax2 = plt.subplot(2,2,2)
    ax2.plot(x, sf(x), "r-")
    ax3 = plt.subplot(2,2,3)
    ax3.plot(x,nf(x), "r-")
    ax4 = plt.subplot(2,2,4)
    ax4.plot(x,twf(x), "r-")
    ax1.scatter(xh,yh)
    ax2.scatter(xh,yh)
    ax3.scatter(xh,yh)
    ax4.scatter(xh,yh)
    ax1.set_title("Degree 3")
    ax2.set_title("Degree 6")
    ax3.set_title("Degree 9")
    ax4.set_title("Degree 12")
    plt.suptitle('Least Squares Solutions of Degree 3, 6, 9, 12')
    plt.show()

    #I am aware that this is probably an awful way of doing this, but I hope it still suffices. I tried to be fancy with defining multiple things at once and it kept breaking.
    return()
    raise NotImplementedError("Problem 3 Incomplete")


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    raise NotImplementedError("Problem 6 Incomplete")

def main():
    m = int(input('Rows: '))
    n = int(input('Columns: '))
    A = np.random.normal(size=(m,n))
    b = np.random.normal(size=(m,1))
    x = least_squares(A,b)
    print(x)

    line_fit()

    polynomial_fit()

main()
