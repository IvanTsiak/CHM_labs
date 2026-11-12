import math

def f(x):
    return x**3 + 3*x**2 - 24*x + 1

def df(x):
    return 3*x**2 + 6*x - 24

eps = 1e-4

print("Метод дихотомії:")
a_n = 0
b_n = 1
x_prev = (a_n + b_n) / 2
iter_d = 1

while True:
    if math.copysign(1, f(a_n)) != math.copysign(1, f(x_prev)):
        b_n = x_prev
    else:
        a_n = x_prev

    x_curr = (a_n + b_n) / 2

    print(f"{iter_d} | {x_curr:.6f} | {f(x_curr):.6f} | {abs(x_curr - x_prev):.6f}")
    if abs(x_curr - x_prev) <= eps:
        print(f"Корінь: {x_curr:.6f}")
        print(f"Кількість ітерацій: {iter_d}")
        break
 
    x_prev = x_curr
    iter_d += 1

print("\nМодифікований метод Ньютона:")
x0 = 0
df_x0 = df(x0)
xn = x0
iter_n = 0

while True:
    iter_n += 1
    xn_next = xn - f(xn) / df_x0

    print(f"{iter_n} | {xn_next:.6f} | {f(xn_next):.6f} | {abs(xn_next - xn):.6f}")
    if abs(xn_next - xn) <= eps:
        print(f"Корінь: {xn_next:.6f}")
        print(f"Кількість ітерацій: {iter_n}")
        break

    xn = xn_next