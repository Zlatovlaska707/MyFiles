import matplotlib.pyplot as plt  # ВЕРСИЯ 3.7.3
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize_scalar

########################################################################################################################
#################################   НАСТРОЙКИ   ########################################################################
start_point = [-1.0, 0.0]  # стартовая точка спусков
koef = [-4, 2, 6]  # коэффициенты для построения "колец" функции
# область рисования(увеличить если ваши кольца не отрисовываются полностью)
x1_min = -0.25
x1_max = 1
x2_min = -1
x2_max = 3

# ВАШ ВАРИАНТ
k1, k2, k3, k4, k5 = [405.45, - 267.3, 49.05, - 10, - 10]

# цвет линии сходимости, можно в формате HEX '#FF0000'
line_color = 'blue'
# цвет линий функции, можно в формате HEX '#FF0000'
f_color = 'green'


########################################################################################################################
########################################################################################################################
def f(x):
    x1 = x[0]
    x2 = x[1]
    return k1 * x1 ** 2 + k2 * x1 * x2 + k3 * x2 ** 2 + k4 * x1 + k5 * x2


gesse = np.matrix([
    [2 * k1, k2],
    [k2, 2 * k3]
])


def grad(x):
    x1 = x[0]
    x2 = x[1]
    return np.array([2 * k1 * x1 + k2 * x2 + k4, k2 * x1 + 2 * k3 * x2 + k5])


def plot_equations(n_values):
    x1 = np.linspace(x1_min, x1_max, 400)
    x2 = np.linspace(x2_min, x2_max, 400)
    x1, x2 = np.meshgrid(x1, x2)

    for n in n_values:
        plt.contour(x1, x2, f((x1, x2)), [n],
                    colors=[f_color])

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)


def one(x):
    trajectory = [x]
    eps = 0.0001

    while len(trajectory) == 1 or np.linalg.norm(trajectory[-1] - trajectory[-2]) > eps:
        e = grad(trajectory[-1])
        r = minimize_scalar(lambda l: f(x - l * e)).x
        x = x - r * e
        trajectory.append(x)

    resx, resy = [], []
    for t in trajectory:
        print(t)
        resx.append(t[0])
        resy.append(t[1])
    return resx, resy


def two(x):
    x0 = np.matrix([[x[0]], [x[1]]])
    trajectory = [x0]
    lu, piv = lu_factor(gesse)
    t_1 = gesse * trajectory[-1]
    t_2 = grad(trajectory[-1]).reshape(-1, 1)
    b = t_1 - t_2
    res = lu_solve((lu, piv), b)
    trajectory.append(np.matrix(res))

    resx, resy = [], []
    for t in trajectory:
        t = t.reshape(-1)
        x, y = t.tolist()[0]
        print([x, y])
        resx.append(x)
        resy.append(y)
    return resx, resy


def main():
    plt.figure(figsize=(10, 8))
    plot_equations(koef)
    resx, resy = one(start_point)
    plt.plot(resx, resy, color=line_color)
    plt.savefig("one.png")
    plt.show()
    # 2
    plt.figure(figsize=(10, 8))
    plot_equations(koef)
    resx, resy = two(start_point)
    plt.plot(resx, resy, color=line_color)
    plt.savefig("two.png")
    plt.show()


if __name__ == '__main__':
    main()
