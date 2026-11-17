import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -np.log(1 - x)

a = 0.0
b = 0.99
num_nodes = 9 

x_nodes = np.linspace(a, b, num_nodes)
y_nodes = f(x_nodes)

print("Вузли інтерполяції:")
print(np.round(x_nodes, 4))
print("Значення функції:")
print(np.round(y_nodes, 4))

def piecewise_linear(x_val, x, y):
    if x_val <= x[0]:
        return y[0]
    for i in range(1, len(x)):
        if x_val <= x[i] or i == len(x) - 1:
            return (y[i - 1] * (x[i] - x_val) + y[i] * (x_val - x[i - 1])) / (x[i] - x[i - 1])

def build_natural_cubic_spline(x, y):
    n_int = len(x) - 1
    h = np.diff(x)
    m = n_int - 1

    lower = np.zeros(m)
    diag = np.zeros(m)
    upper = np.zeros(m)
    rhs = np.zeros(m)

    for k in range(m):
        i = k + 1
        h_i = h[i - 1]
        h_next = h[i]

        if k > 0:
            lower[k] = h_i
        diag[k] = 2.0 * (h_i + h_next)
        if k < m - 1:
            upper[k] = h_next

        rhs[k] = 6.0 * ((y[i + 1] - y[i]) / h_next - (y[i] - y[i - 1]) / h_i)

    for k in range(1, m):
        w = lower[k] / diag[k - 1]
        diag[k] -= w * upper[k - 1]
        rhs[k] -= w * rhs[k - 1]

    c_inner = np.zeros(m)
    c_inner[m - 1] = rhs[m - 1] / diag[m - 1]
    for k in range(m - 2, -1, -1):
        c_inner[k] = (rhs[k] - upper[k] * c_inner[k + 1]) / diag[k]

    c = np.zeros(n_int + 1)
    c[1:n_int] = c_inner

    a_coef = np.zeros(n_int + 1)
    b_coef = np.zeros(n_int + 1)
    d_coef = np.zeros(n_int + 1)

    for i in range(1, n_int + 1):
        h_i = h[i - 1]
        a_coef[i] = y[i]
        d_coef[i] = (c[i] - c[i - 1]) / h_i
        b_coef[i] = (h_i / 2.0) * c[i] - (h_i ** 2 / 6.0) * d_coef[i] + (y[i] - y[i - 1]) / h_i

    return a_coef, b_coef, c, d_coef

a_c, b_c, c_c, d_c = build_natural_cubic_spline(x_nodes, y_nodes)

def evaluate_spline(x_val, x, a_coef, b_coef, c_coef, d_coef):
    if x_val <= x[0]:
        dx = x[0] - x[1]
        return a_coef[1] + b_coef[1] * dx + (c_coef[1] / 2.0) * (dx ** 2) + (d_coef[1] / 6.0) * (dx ** 3)
    for i in range(1, len(x)):
        if x_val <= x[i] or i == len(x) - 1:
            dx = x_val - x[i]
            return a_coef[i] + b_coef[i] * dx + (c_coef[i] / 2.0) * (dx ** 2) + (d_coef[i] / 6.0) * (dx ** 3)

x_grid = np.linspace(a, b, 1000)
y_exact = f(x_grid)
y_lin = np.array([piecewise_linear(val, x_nodes, y_nodes) for val in x_grid])
y_spl = np.array([evaluate_spline(val, x_nodes, a_c, b_c, c_c, d_c) for val in x_grid])

err_lin = np.abs(y_exact - y_lin)
err_spl = np.abs(y_exact - y_spl)

print(f"\nМаксимальна похибка кусково-лінійної інтерполяції: {np.max(err_lin)}")
print(f"Максимальна похибка кубічного сплайна: {np.max(err_spl)}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x_grid, y_exact, label='y(x) = -ln(1 - x)', color='blue', linewidth=2)
plt.plot(x_grid, y_lin, label='L(x)', color='green', linestyle='--')
plt.plot(x_grid, y_spl, label='s(x)', color='red', linestyle='-.')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Вузли (n=9)')
plt.title('Графіки y(x), L(x) та s(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_grid, err_lin, label='|y(x) - L(x)|', color='green', linewidth=2)
plt.title('Похибка |y(x) - L(x)|')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_grid, err_spl, label='|y(x) - s(x)|', color='red', linewidth=2)
plt.title('Похибка |y(x) - s(x)|')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()