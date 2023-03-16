import numpy as np

##############################
# Part 1
##############################
# Write your answers to the math problems below. To represent exponents, use the ^ symbol.
# 1.
#
# 2.
#
# 3.
#
# 4.
#

##############################
# Part 2
##############################

def problem1 (A, B, C):
    return np.dot(A, B)-C

def problem2 (A):
    return np.ones(np.shape(A))

def problem3 (A):
    np.fill_diagonal(A, 0)
    return A

def problem4 (A, i):
    return A[i].sum()

def problem5 (A, c, d):
    return np.mean(A[c:d+1])

def problem6 (A, k):
    w, v = np.linalg.eig(A)
    return v.max(k)

def problem7 (A, x):
    #TODO finish
    print(A)
    return np.linalg.solve(A, x)

def problem8 (x, k):
    return np.tile(x,(k, 1))

def problem9 (A):
    B = A
    B = np.random.permutation(B)
    return B

def problem10 (A):
    return np.mean(A, axis=1)


def main():
    A = np.array([[1,2,3],[2,3,4],[3,4,5]])
    print(problem7(A,3))

if __name__ == "__main__":
    main()