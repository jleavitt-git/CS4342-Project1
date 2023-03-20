import numpy as np

##############################
# Part 1
##############################
# Write your answers to the math problems below. To represent exponents, use the ^ symbol.
# 1.
#   dh/dx = y + z + s6(z^2), dh/dy
#   dh/dy = x + 2x(z^2)
#   dh/dz = x + 4xyz
# 2.
#   g/du = log (v) + 0 = log(v)
#    dg/dv =
#           a: (d/du) ulog(v) = (d/du) uln(v) = u/v
#           b: (d/du) e^(v^2) = 2v*e^(v^2) = 2v(e^(v^2)
#  Solution: dg/du = log (v) 
#            dh/dv = (u/v) + 2v(e^(v^2))
#
# 3. Given A = 5x6 and C is 4x9, B^T = 6x4, so 
#    B = 4x6
#
# 4.
#    (p^T) = [1 -1 -3]
#    (p^T)Q = [-1 -2]
#    (p^T)Qr = [-x-2y]
#  Solution: dg/dx = -1
#            dg/dy = -2

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
    return np.mean((A >= c) & (A <= d))

def problem6 (A, k):
    if k <= len(A):
        vals, vects = np.linalg.eig(A)
        maxIndexes = np.argpartition(vals, -k)[-k:]
        maxVectors = np.zeros(shape=(k, len(A)))
        for i in range(len(maxIndexes)):
            maxVectors[i] = vects[:,maxIndexes[i]]
        return maxVectors
    else:
        return ("Impossible")

def problem7 (A, x):
    return np.linalg.solve(A, x)

def problem8 (x, k):
    return np.tile(x,(k, 1))

def problem9 (A):
    B = A.copy()
    B = np.random.permutation(B)
    return B

def problem10 (A):
    return np.mean(A, axis=1)


def main():
    A = np.array([[1,2,3],[2,3,4],[3,4,5]])
    print(problem7(A,3))

if __name__ == "__main__":
    main()