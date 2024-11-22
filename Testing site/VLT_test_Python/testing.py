import numpy as np

def E(i, j, size):
    E = np.eye(size)
    E[i, i], E[j, j] = 0, 0
    E[i, j], E[j, i] = 1, 1
    return E

def S(i, j, λ, size):
    S = np.eye(size)
    S[i, j] = λ
    return S

def M(i, λ, size):
    M = np.eye(size)
    M[i, i] = λ
    return M

def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    m, n = A.shape
    T = []

    for j in range(min(m, n)):
        # Find pivot element
        pivot_row = j
        for i in range(j, m):
            if abs(A[i, j]) > abs(A[pivot_row, j]):
                pivot_row = i

        # Swap rows if necessary
        if A[pivot_row, j] != 0:
            if pivot_row != j:
                A[[j, pivot_row]] = A[[pivot_row, j]]
                b[[j, pivot_row]] = b[[pivot_row, j]]
                T.append(E(j, pivot_row, m))

            # Normalize pivot row
            leading_entry = A[j, j]
            if leading_entry != 1:
                A[j] = A[j] / leading_entry
                b[j] = b[j] / leading_entry
                T.append(M(j, 1 / leading_entry, m))

            # Eliminate elements below and above the pivot
            for i in range(m):
                if i != j and A[i, j] != 0:
                    λ = -A[i, j]
                    A[i] += λ * A[j]
                    b[i] += λ * b[j]
                    T.append(S(i, j, λ, m))

    return T, A, b

# Example usage:
A = np.array([[1, 2, 3], [0, 0, 1], [0, 0, 1]])
b = np.array([[0], [0], [0]])

T, A_reduced, b_reduced = gaussian_elimination(A, b)

print("Transformation matrices T:")
for t in T:
    print(t)

print("Reduced matrix A:")
print(A_reduced)

print("Reduced vector b:")
print(b_reduced)