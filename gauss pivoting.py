import numpy as np


def back_substitution(U, c):
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = c[i]
        for j in range(i + 1, n):
            s = s - U[i][j] * x[j]
        x[i] = s / U[i][i]
    return x


def gauss_partial(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    for k in range(n - 1):
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i

        if A[p][k] == 0:
            raise ValueError("No unique solution")

        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] = A[i][j] - m * A[k][j]
            b[i] = b[i] - m * b[k]

    return back_substitution(A, b)


def gauss_total(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    order = list(range(n))

    for k in range(n - 1):
        p, q = k, k
        for i in range(k, n):
            for j in range(k, n):
                if abs(A[i][j]) > abs(A[p][q]):
                    p, q = i, j

        if A[p][q] == 0:
            raise ValueError("No unique solution")

        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]
        A[:, [k, q]] = A[:, [q, k]]
        order[k], order[q] = order[q], order[k]

        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] = A[i][j] - m * A[k][j]
            b[i] = b[i] - m * b[k]

    y = back_substitution(A, b)
    x = np.zeros(n)
    for k in range(n):
        x[order[k]] = y[k]
    return x


A = [[1, 2, 3],
     [3, 4, 1],
     [2, 1, 2]]
b = [14, 14, 10]

print("Partial pivoting:", gauss_partial(A, b))
print("Total pivoting  :", gauss_total(A, b))
print("NumPy reference :", np.linalg.solve(A, b))

A2 = [[1e-20, 1], [1, 1]]
b2 = [1, 2]
print("Tiny pivot, partial:", gauss_partial(A2, b2))
print("Tiny pivot, total  :", gauss_total(A2, b2))
