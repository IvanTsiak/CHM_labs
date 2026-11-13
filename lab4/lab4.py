import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def f(x):
    return -np.log(1 - x)

a = 0.0
b = 0.99
n = 5
h = (b - a) / n

x_nodes = np.array([a + i * h for i in range(n + 1)])
y_nodes = f(x_nodes)

print("Вузли інтерполяції x:", np.round(x_nodes, 4))
print("Значення функції у вузлах y:", np.round(y_nodes, 4))

def divided_differences(x, y):
    m = len(x)
    coef = np.zeros([m, m])
    coef[:, 0] = y
    
    for j in range(1, m):
        for i in range(m - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

newton_coef = divided_differences(x_nodes, y_nodes)

def newton_poly(x_val, x, coef):
    m = len(x)
    p = coef[0]
    product = 1.0
    for i in range(1, m):
        product *= (x_val - x[i-1])
        p += coef[i] * product
    return p

x_grid = np.linspace(0, 0.99, 500)
y_exact = f(x_grid)
y_approx = np.array([newton_poly(val, x_nodes, newton_coef) for val in x_grid])
error = np.abs(y_exact - y_approx)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(x_grid, y_exact, label='y(x) = -ln(1 - x)', color='blue', linewidth=2)
plt.plot(x_grid, y_approx, label=f'Pn(x) (степінь {n})', color='red', linestyle='--', linewidth=2)
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Вузли інтерполяції')
plt.title('Порівняння функції та полінома Ньютона')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_grid, error, label='|y(x) - Pn(x)|', color='purple', linewidth=2)
plt.title('Модуль похибки інтерполяції')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

y_target = 1.5

def equation_to_solve(x):
    return newton_poly(x, x_nodes, newton_coef) - y_target

x_star_direct = brentq(equation_to_solve, 0, 0.99)
print(f"\nПряма інтерполяція: знайдений x* для y* = 1.5: {x_star_direct:.6f}")

inv_coef = divided_differences(y_nodes, x_nodes)

def inverse_newton_poly(y_val, y, coef):
    m = len(y)
    p = coef[0]
    product = 1.0
    for i in range(1, m):
        product *= (y_val - y[i-1])
        p += coef[i] * product
    return p

x_star_inverse = inverse_newton_poly(y_target, y_nodes, inv_coef)
print(f"Обернена інтерполяція: знайдений x* для y* = 1.5: {x_star_inverse:.6f}")