# ndarray.ndim = the number of axes
# ndarray.shape = size of the array
# ndarray.size = total number of elements
# ndarray.dtype = data type
# ndarray.itemsize = the size of each element

from numpy import *

#creation
a = arange(15).reshape(3,5)
print a

b = array([6,7,8])
c = zeros((3,5))
d = ones((2,3,4), dtype = int16)
e = linspace(0,2,9) # 9 numbers from 0 to 2
print e

f = arange(0,4,0.5)
print f

print a.dtype
print a.dtype.name

print a.itemsize
