import numpy as np

A = np.array([
    [3.0, 1.0, 0.0, 0.0],
    [1.0, 3.5, -2.0, 0.0],
    [0.0, -2.0, 4.0, 1.2],
    [0.0, 0.0, 1.2, 1.5]
])
eps = 1e-4
n = A.shape[0]

def vector_norm_inf(v):
    max_val = 0
    for x in v:
        if abs(x) > max_val:
            max_val = abs(x)
    return max_val

def matrix_norm_inf(mat):
    max_sum = 0
    for i in range(mat.shape[0]):
        row_sum = sum(abs(x) for x in mat[i])
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum

def power_method(mat, epsilon):
    x_k = np.ones(mat.shape[0])
    lambda_k = 0
    while True:
        x_new = np.dot(mat, x_k)
        
        norm_val = vector_norm_inf(x_new) 
        
        lambda_new = x_new[0] / x_k[0] 
        x_k = x_new / norm_val 
        
        if np.abs(lambda_new - lambda_k) <= epsilon:
            return lambda_new
        lambda_k = lambda_new

lambda_max = power_method(A, eps)

norm_inf_A = matrix_norm_inf(A)
B = norm_inf_A * np.eye(n) - A
lambda_max_B = power_method(B, eps)
lambda_min = norm_inf_A - lambda_max_B

print("Степеневий метод")
print(f"Максимальне власне значення: {lambda_max:.4f}")
print(f"Мінімальне власне значення: {lambda_min:.4f}\n")

def jacobi_eigen(mat, epsilon):
    A_k = mat.copy()
    U = np.eye(n)
    
    while True:
        max_val = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_k[i, j]) > max_val:
                    max_val = abs(A_k[i, j])
                    p, q = i, j
                    
        off_diag_sum = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag_sum += A_k[i, j] ** 2
                    
        if off_diag_sum <= epsilon:
            break
            
        if A_k[p, p] == A_k[q, q]:
            phi = np.pi / 4 if A_k[p, q] > 0 else -np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A_k[p, q] / (A_k[p, p] - A_k[q, q]))
            
        U_k = np.eye(n)
        U_k[p, p] = np.cos(phi)
        U_k[q, q] = np.cos(phi)
        U_k[p, q] = -np.sin(phi)
        U_k[q, p] = np.sin(phi)
        
        A_k = np.dot(U_k.T, np.dot(A_k, U_k))
        U = np.dot(U, U_k)
        
    eigenvalues = np.diag(A_k)
    eigenvectors = U
    return eigenvalues, eigenvectors

eigenvalues_jacobi, eigenvectors_jacobi = jacobi_eigen(A, eps)

print("Метод Якобі")
print("Власні значення:")
print(np.round(eigenvalues_jacobi, 4))
print("Власні вектори:")
print(np.round(eigenvectors_jacobi, 4))