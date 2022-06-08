# python_intro.py
"""Python Essentials: Introduction to Python.
Aidan Wittman
MTH420 Prof. Gibson Spring 2022
4/8/2022
"""

import numpy as np

# Problem 1 asks to compute the volume of a sphere with radius 10. This is the associated formula.

pi = 3.14159
ft = 4/3
r = 10
print(pi*ft*r**3)

# Problem 2 just asks to print Hello, world! through Terminal, so I'm assuming that's not necessary in the final project. Just for old time's sake, I'll throw in a "Hello, world!" though."""
print("Hello, world!")

# Problem 3

def sphere_volume(r):
	"""This function takes some input r and calculates the volume of a sphere with radius r."""
	pi = 3.14159
	fourthirds = 4/3
	volume = pi * fourthirds * r**3
	return(volume)

r = input("Enter the radius of a sphere: ")                      	# This line asks for an input from the user and stores it as a string.
r = float(r)                                                     	# We want the input to be a float, so we convert the string into a float.
x = sphere_volume(r)                                             	# Save a variable x as the result of the function sphere_volume.
print("The volume of a sphere with radius ", r, " is, ", x, ".") 	# Print the volume of the sphere.

# Problem 4

def prob4():
	"""This function takes two pre-determined matrices A and B and returns their product AB as C."""
	A = np.array([[3, -1, 4], [1, 5, -9]])
	B = np.array([[[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]]])
	C = np.dot(A, B)
	return(C)
    
x = prob4()
print(x)

# Problem 5

def tax_liability(income):
    """This function takes an income and then calculates te tax liability for that income using the associated tax bracket table.
    """
    if (income > 40125):
        income = income - 40125
        tax = (9875 * 0.1) + (30249.99 * .12) + income * 0.22
        return(tax)
    elif ((income <= 40125) & (income > 9875)):
        income = income - 9875
        tax = (9875 * 0.1) + income * 0.12
        return(tax)
    else:
        tax = income * 0.1
        return(tax)

income = input("Please enter your income: ")
income = float(income)
tax = tax_liability(income)
print("Your tax liability with an income of $", income, " is $", tax, ".")

# Problem 6

def prob6a():
	"""This function performs three operations on the pre-determined matrices A and B defined as lists and returns the results of those three operations."""
	A = [1, 2, 3, 4, 5, 6, 7]
	B = [5, 5, 5, 5, 5, 5, 5]
	mult = [a * b for a, b in zip(A, B)]
	ad = [a + b for a, b in zip(A, B)]
	scal = [5 * a for a in A]
	return(mult, ad, scal)
    
def prob6b():
	"""This function performs three operations on the pre-determined matrices A and B as numpy matrices and returns the results of those three operations."""
	A = np.array([1, 2, 3, 4, 5, 6, 7])
	B = np.array([5, 5, 5, 5, 5, 5, 5])
	multiply = A * B
	add = A + B
	scalar = 5 * A
	return(multiply, add, scalar)

u, v, w = prob6a()
print("A * B:", u, "\nA + B:", v, "\n5A:", w)

x, y, z = prob6b()
print("\n\nA * B:", x, "\nA + B:", y, "\n5A:", z)
