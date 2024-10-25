#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:23:18 2023

@author: konuci
"""
import numpy as np
import math
print("Ex.1\n")
k = 100.

one = [0.7, 0.5, 0.8] 
two = [0.25, 0.75, 0.5]

#perturbation without closure
p1=[x * y for x, y in zip(one, two)]
p1_1 = [k/sum(p1)*p1[i] for i in range(3)]
#perturbation with closure

one_cl = []
two_cl = []

for i in one:
    one_cl.append(i * k / sum(one))
    
for j in two:
    two_cl.append(j *k / sum(two))

p2 = [x * y for x, y in zip(one_cl, two_cl)]
    
p2_2 = [k/sum(p2) * p2[i] for i in range(3)]

print(p1_1, p2_2,"\n")

# The result is the same, whether or not you close before perturbing.

print("Ex.2\n")

x = [0.7, 0.4, 0.8]
y = [2., 8., 1.]

aip = 1./(2*3) * sum([np.log(x[i]/x[j])*np.log(y[i]/y[j])
                      for i in range(3) for j in range(3)])

print(aip,"\n")

# -0.77; not orthogonal. Orthogonal vectors have a dot product equal to 0

print("Ex.3\n")

x = [3.74, 9.35, 16.82, 18.69, 23.36, 28.04]
y = [9.35, 28.04, 16.82, 3.74, 18.69, 23.36]

# Aitchison norm
an_x = math.sqrt(1./(2*len(x)) *
               sum([sum([(math.log(i/j)**2) for i in x]) for j in x]))
an_y = math.sqrt(1./(2*len(y)) *
               sum([sum([(math.log(i/j)**2) for i in y]) for j in y]))
# Aichison inner product
aip = 1./(2*len(x)) * sum([sum([np.log(x[i]/x[j])*np.log(y[i]/y[j])
                                for i in range(6)]) for j in range(6)])

angle = math.acos((aip)/(an_x*an_y))

print(angle,"\n")

print("Ex.4\n")

x = [0.7, 0.4, 0.8]

a = math.sqrt(1./(2.*len(x)) *
               sum([sum([(math.log(i/j)**2) for i in x]) for j in x]))
alfa = 1./a

res = [(x[i]**alfa) for i in range(3)]

anorm = math.sqrt(1./(2.*len(res)) *
               sum([sum([(math.log(res[i]/res[j])**2) for i in range(3)]) for j in range(3)]))

print(anorm,"\n")

# 1.; We have performed a generalized multiplication of a composition and its
# inverse norm, that is, we have (generalized) divided the composition with its
# length, which is to normalize the vector. The length (norm) of a normalized vector
# is unity.

print("Ex.5\n")

x1 = [79.07, 12.83, 8.10]
x2 = [31.74, 56.69, 11.57]

distance = math.sqrt(1./(2*len(x1)) *
               sum([sum([((math.log(x1[i]/x1[j]) - math.log(x2[i]/x2[j]))**2) for i in range(3)]) for j in range(3)]))
   
print(distance)

                                                                                                                
x1_95 = [95./sum(x1) * x1[i] for i in range(3)]
x2_95 = [95./sum(x2) * x2[i] for i in range(3)]

x1 = x1_95+[5]
x2 = x2_95+[5]           

distance = math.sqrt(1./(2*len(x1)) *
               sum([sum([((math.log(x1[i]/x1[j]) - math.log(x2[i]/x2[j]))**2) for i in range(4)]) for j in range(4)]))

print(distance)

	

print("Ex.6\n")

data = np.array([[79.07, 12.83, 8.10], [31.74, 56.69, 11.57],
                 [18.61, 72.05, 9.34], [49.51, 15.11, 35.38],
                 [29.22, 52.36, 18.42]])

# First calculate the geometric means of the compositions
gm = [pow(data[i][0]*data[i][1]*data[i][2], 1./3) for i in range(5)]
clr = [[np.log(data[i][j]/gm[i]) for j in range(3)] for i in range(5)]
print(np.round(clr, 2))
print(np.round([sum(clr[i]) for i in range(5)], 2))


print("Ex.7\n")
import matplotlib.pyplot as plt

data = np.array([[79.07, 12.83, 8.10], [31.74, 56.69, 11.57],
                 [18.61, 72.05, 9.34], [49.51, 15.11, 35.38],
                 [29.22, 52.36, 18.42]])

# use the third part as denominator for an ALR tranformation
alr = [[np.log(data[i][0]/data[i][2]), np.log(data[i][1]/data[i][2])]
       for i in range(5)]
print(alr)

_ = [plt.plot([alr[i][0]], [alr[i][1]], 'o', color='maroon') for i in range(5)]

# now use the first part as denominator for an ALR tranformation
alr = [[np.log(data[i][1]/data[i][0]), np.log(data[i][2]/data[i][0])]
       for i in range(5)]

print(alr)
_ = [plt.plot([alr[i][0]], [alr[i][1]], 'o', color='steelblue')
     for i in range(5)]

# and finally the second part as denominator for an ALR tranformation
alr = [[np.log(data[i][0]/data[i][1]), np.log(data[i][2]/data[i][1])]
       for i in range(5)]

print(alr)
_ = [plt.plot([alr[i][0]], [alr[i][1]], 'o', color='seagreen')
     for i in range(5)]

plt.show()

print("Ex.8\n")

x1 = [math.e,1,1]
x2 = [1,math.e,1]

#aichision inne product

aip = 1./(2*len(x1)) * sum([sum([np.log(x1[i]/x1[j])*np.log(x2[i]/x2[j])
                                for i in range(len(x1))]) for j in range(len(x2))])
if aip == 0:
    print(aip,"orthogonal")
else:
    print(aip, "not orthogonal")

print("Ex.9\n")

ptb = [[1,-1,0], [1,1,-1]]
normalized = [[1./1*math.sqrt(1./2), -1./1*math.sqrt(1./2), 0],
        [1./2*math.sqrt(2./3), 1./2*np.sqrt(2./3), -1./1*math.sqrt(1./3)]]


print('\nExercise 3.10')
# Use data from Exercise 2.3

data = np.array([[79.07, 12.83, 8.10], [31.74, 56.69, 11.57],
                 [18.61, 72.05, 9.34], [49.51, 15.11, 35.38],
                 [29.22, 52.36, 18.42]])

gm = [pow(data[i][0]*data[i][1]*data[i][2], 1./3) for i in range(5)]
clr = [[np.log(data[i][j]/gm[i]) for j in range(3)] for i in range(5)]

# Use ptbn from previous exercise
ilr = [np.dot(np.array(clr[i]), np.array(normalized).T) for i in range(5)]
_ = [plt.plot([ilr[i][0]], [ilr[i][1]], 'x', color='blue') for i in range(5)]

# Using another binary partion basis
ptb = [[1, 1, -1], [1, -1, 0]]

# Then we nomalize it using formula 3.22 in the lecture notes:
ptbn = [[1./2*np.sqrt(2./3), 1./2*np.sqrt(2./3), -1./1*np.sqrt(2./3)],
        [1./1*np.sqrt(1./2), -1./1*np.sqrt(1./2), 0]]

ilr = [np.dot(np.array(clr[i]), np.array(ptbn).T) for i in range(5)]
_ = [plt.plot([ilr[i][0]], [ilr[i][1]], 'x', color='red') for i in range(5)]

# Notice that the two sets of ILR coordinates are equally interspaced, just flipped on both axis
# The ALR coordinates does not have this quality.




