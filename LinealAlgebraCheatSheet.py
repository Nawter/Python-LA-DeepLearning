# From https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c
import numpy as np
# Multiply two arrays
x = [1, 2, 3]
y = [3, 5, 7]
product = []
print("First vector:\n",x)
print("Second vector:\n",y)
for i in range(len(x)):
    product.append(x[i]*y[i])
print('-------------Manual-Version------------')
print(product)
# Lineal algebra version
x = np.array([1, 2, 3])
y = np.array([3, 5, 7])
print('-------------Lineal-Algebra-Version---------')
# Elementwise operations like addition, subtraction, and division.
print("product", x * y)
print("addition", y + x)
print("subtraction", y - x)
print("division", y / x)
# Dot product of two vectos is a scalar.
print("Dot product:", np.dot(x,y))

# Matrix dimensions
a = np.array([
    [1,3,5],
    [7,9,11]
])
print("Dimension of a:",a.shape)
print("First matrix:\n",a)
b = np.array([
    [1,3],
    [5,7],
    [9,11]
])
print("Dimension of b:",b.shape)
print("Second matrix:\n",b)
c = np.array([
    [1,1,1],
    [2,2,2]
])
print("Dimension of c:",c.shape)
print("Third matrix:\n",c)
# Matrix scalar operations
print("Matrix scalar operation add one to a:\n",a + 1)

# Matrix elementwise operations
print("Add two matrix:\n",a + c)

# Broadcasting in numpy
m1 = np.array([
    [1],
    [3]
])
print("First matrix:\n",m1)
m2 = np.array([
    [5,7],
    [9,13]
])
print("Second matrix:\n",m2)
m3 = np.array([
     [1,3]
])
print("Third matrix:\n",m3)
# Same no. of rows
# Different no. of columns
# but m1 has one column so this works
print("Broadcasting same number of rows:\n",m1 * m2)

# Same no. of colums
# Different no. of rows
# but m3 has one column so this works
print("Broadcasting same number of colums:\n",m2 * m3)

# Different no. of rows
# Different no. of columns
# but m1 has one column so this works
print("Broadcasting different number of rows and columns:\n",m1 * m3)

# Matrix hadamard product
m4 = np.array([
    [2,3],
    [2,3]])
print("First matrix:\n",m4)
m5 = np.array([
    [3,4],
    [5,6]
])
print("Second matrix:\n",m5)
print("Matrix hadamard product:\n",m4*m5)

# Matrix transpose.
m6 = np.array([
    [1, 2],
    [3, 4]])
print("Original matrix:\n",m6)
print("Matrix transpose:\n",m6.T)

# Matrix multiplication using numpy
m7 = np.array([
    [1, 2]
])
print("First matrix:\n",m7)
print("Dimension of m7:\n",m7.shape)
m8 = np.array([
    [3, 5],
    [7, 9]
])
print("Second matrix:\n",m8)
print("Dimension of m8:\n",m8.shape)
mm = np.dot(m7,m8)
print("Multiply two matrices:\n", mm)
print("Dimension of the final matrix:\n", mm.shape)
