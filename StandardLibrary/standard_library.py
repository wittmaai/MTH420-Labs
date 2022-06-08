# standard_library.py
"""Python Essentials: The Standard Library.
Aidan Wittman
MTH420 Prof. Gibson Spring 2022
5/30/2022
"""

import calculator

from itertools import combinations

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return((min(L),max(L),(sum(L)/len(L))))

    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    print("The mutable objects types are: list, set. \nThe immutable object types are: int, str, tuple.")
    return()
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    a = calculator.product(a,a)
    b = calculator.product(b,b)
    w = calculator.sum(a,b)
    w = calculator.squarey(w)
    return(w)
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    m = len(A)
    mathylist = []
    for i in range(m+1):
        mathylist += list(combinations(A, i))
        i += 1
    return(mathylist)
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""

def main():
    elements = []
    n = int(input("You will enter a list of numbers. Please state how many elements will be in this list." ))
    for i in range(n):
        nele = int(input(f"Number {i+1}: "))
        elements.append(nele)
        i += 1
    print(elements)
    x, y, z = prob1(elements)
    print("Minimum:", x, "Maximum:", y, "Average", z)
    prob2()
    print("You will enter two sides of a triangle to calculate its hypotenuse.")
    x = int(input("First side: "))
    y = int(input("Second side: "))
    z = hypot(x,y)
    print("The hypotenuse of your triangle is: ", z)
    mathyset = str(input("Please enter a set. I will return its power set. (For example: abc): "))
    powerset = power_set(mathyset)
    print(powerset)
main()
