# python_intro.py
"""Python Essentials: Introduction to Python.
Aidan Wittman
MTH420 Prof. Gibson Spring 2022
5/30/2022
"""

import numpy as np

#Problem 1
def isolate(a, b, c, d, e):
    """Receive 5 character inputs. Return those inputs
    with three spaces between the first two, and one
    space between the last two. This is accomplished using
    the built-in sep and end functions within the print function.
    """
    print(a, b, c, sep="     ", end=" ")
    print(d, e)
    return()
    raise NotImplementedError("Problem 1 Incomplete")

#Problem 2
def first_half(string):
    """Receive a string. Print the first half of that string.
    """
    firsthalf = string[:len(string)//2]
    #print('Here is the first half:', string[:len(string)//2])
    return(firsthalf)
    raise NotImplementedError("Problem 2 Incomplete")


def backward(first_string):
    """Receive a string. Print that string backwards.
    """
    backwards = first_string[len(first_string)-1::-1]
    #print('Here is it backwards:', first_string[len(first_string)-1::-1])
    return(backwards)
    raise NotImplementedError("Problem 2 Incomplete")

#Problem 3
def list_ops():
    """Start with a list of bear, ant, cat, dog.
    Append eagle.
    Remove (or pop) the entry at index 1.
    Sort the list in reverse alphabetical order.
    Repalce eagle with hawk. (index()).
    Add hunter to the last entry in the list.
    Return the resulting list.
    """
    animals = ["bear", "ant", "cat", "dog"]
    animals[2] = "fox"
    animals.append("eagle")
    animals.pop(1)
    animals.sort()
    animals.reverse()
    animals[1] = "hawk"
    animals[3] = animals[3] + "hunter"
    #print(animals) #This should be [fox, hawk, dog, bearhunter].
    return(animals)
    raise NotImplementedError("Problem 3 Incomplete")

#Problem 4
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    altharmonic = [((-1)**(i+1))/i for i in range(1,n+1)]
    print(sum(altharmonic))
    return()
    raise NotImplementedError("Problem 4 Incomplete")



def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    Acopy = np.copy(A)
    mask = Acopy < 0
    Acopy[mask] = 0
    #print(Acopy)
    return(Acopy)
    raise NotImplementedError("Problem 5 Incomplete")

def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0,2,4],[1,3,5]])
    B = np.array([[3,0,0],[3,3,0],[3,3,3]])
    C = np.array([[-2,0,0],[0,-2,0],[0,0,-2]])
    I = np.eye(3)
    AT = np.transpose(A)
    zero1 = np.zeros((3,3))
    zero2 = np.zeros((2,2))
    zero3 = np.zeros((2,3))
    zero4 = np.zeros((3,2))
    row1 = np.hstack((zero1,AT,I))
    row2 = np.hstack((A,zero2,zero3))
    row3 = np.hstack((B,zero4,C))
    final = np.vstack((row1,row2,row3))
    #print(final)
    return(final)
    raise NotImplementedError("Problem 6 Incomplete")

def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    raise NotImplementedError("Problem 7 Incomplete")


def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    raise NotImplementedError("Problem 8 Incomplete")


def main():
    print('You will enter 5 characters. I will return an oddly separated version of those characters.')
    a = input('Character 1: ')
    b = input('Character 2: ')
    c = input('Character 3: ')
    d = input('Character 4: ')
    e = input('Character 5: ')
    isolate(a,b,c,d,e)

    prob2string = str(input('Please input a string. We will do some stuff with it: '))
    fh = first_half(prob2string)
    b = backward(prob2string)
    print("First Half:", fh, "\nBackward:", b, "\n")

    thing = list_ops()
    print('THING!!!', thing)

    altharmonicnumber = int(input('Enter the number of terms of the alternating harmonic series you would like to sum: '))
    alt_harmonic(altharmonicnumber)

    prob5matrix = np.array([[1,2,-3],[4,5,-6],[-7,8,-9]])
    A = prob5(prob5matrix)
    print(A)

    monster = prob6()
    print(monster)
main()
