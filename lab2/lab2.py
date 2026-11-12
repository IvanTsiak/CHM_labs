import numpy as np

n = 5
b = np.ones(n)

A = np.zeros((n, n))
for i in range(n):
    for j in range (n):
        i_true, j_true = i+1, j+1
        if i_true == n - j_true + 1:
            A[i, j] = n
        else:
            A[i, j] = 1 / (i_true + (n - j_true + 1))

A_bar = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        i_true, j_true = i+1, j+1
        if i_true == j_true:
            A_bar[i, j] = n
        else:
            A_bar[i, j] = 1 / (i_true + j_true)

def gauss_row_pivoting(A_mat, b_vec):
    size = len(b_vec)
    aug_A = A_mat.copy()
    aug_b = b_vec.copy()
    var_idx = np.arange(size)
    p = 0

    for k in range(size - 1):
        max_col = np.argmax(np.abs(aug_A[k, k:size])) + k
        if max_col != k:
            aug_A[:, [k, max_col]] = aug_A[:, [max_col, k]]
            var_idx[k], var_idx[max_col] = var_idx[max_col], var_idx[k]
            p += 1
        for i in range(k + 1, size):
            factor = aug_A[i, k] / aug_A[k, k]
            aug_A[i, k:] -= factor * aug_A[k, k:]
            aug_b[i] -= factor * aug_b[k]

    det = ((-1)**p) * np.prod(np.diag(aug_A))

    y = np.zeros(size)
    for i in range(size - 1, -1, -1):
        y[i] = (aug_b[i] - np.dot(aug_A[i, i+1:], y[i+1:])) / aug_A[i, i]

    x = np.zeros(size)
    for i in range(size):
        x[var_idx[i]] = y[i]

    return x, det

x_gauss, det_A = gauss_row_pivoting(A, b)
A_inv = np.linalg.inv(A)
norm_A = np.linalg.norm(A, np.inf)
norm_A_inv = np.linalg.norm(A_inv, np.inf)
cond_A = norm_A * norm_A_inv
r_gauss = np.dot(A, x_gauss) - b

def jacobi_method(A_mat, b_vec, eps):
    size = len(b_vec)
    x_k = np.zeros(size)

    converges = True
    for i in range(size):
        diag_val = np.abs(A_mat[i, i])
        sum_others = np.sum(np.abs(A_mat[i, :])) - diag_val
        if diag_val < sum_others:
            converges = False
            break

    iterations = 0
    while True:
        x_new = np.zeros(size)
        for i in range(size):
            s = sum(A_mat[i, j] * x_k[j] for j in range(size) if j != i)
            x_new[i] = (b_vec[i] - s) / A_mat[i, i]

        iterations += 1
        if np.max(np.abs(x_new - x_k)) <= eps:
            break
        x_k = x_new

    return x_new, iterations, converges

x_jacobi, iters_jacobi, is_convergent = jacobi_method(A_bar, b, eps=1e-5)
r_jacobi = np.dot(A_bar, x_jacobi) - b

print("Метод Гаусса")
print(f"Розв'язок x: {x_gauss}")
print(f"Визначник: {det_A:.6f}")
print(f"Число обумовленості: {cond_A:.6f}")
print("Обернена матриця:")
print(np.round(A_inv, 4))
print("A^-1 * A:")
print(np.round(np.dot(A_inv, A), 4))
print(f"Нев'язка: {np.round(r_gauss, 10)}\n")

print("Метод Якобі")
print(f"Достатня умова збіжності виконується: {is_convergent}")
print(f"Розв'язок x: {x_jacobi}")
print(f"Кількість ітерацій: {iters_jacobi}")
print(f"Нев'язка: {np.round(r_jacobi, 10)}")