import matplotlib.pyplot as plt  # ВЕРСИЯ 3.7.3 !!!!!!!!
import numpy as np

########################################################################################################################
#################################   НАСТРОЙКИ   ########################################################################
start_point = [-1.0, 0.0]  # стартовая точка спусков
koef = [-4, 2, 6]  # коэффициенты для построения "колец" функции
# область рисования(увеличить если ваши кольца не отрисовываются полностью)
x1_min = -1.1
x1_max = 0.7
x2_min = -0.5
x2_max = 2.5

# ВАШ ВАРИАНТ
k1, k2, k3, k4, k5 = [845.45, - 386.1, 53.25, - 13.69, - 13.69]
# ФУНКЦИЯ ОТ МАТРИЦЫ ИЗ ПЗ9 ТАМ ГДЕ ИНТЕГРАЛ (часть 1 пункт 3)
f_A = np.array([[0.00289482, 0.0101102],
                [0.0101098, 0.0443726]])

# цвет линии сходимости, можно в формате HEX '#FF0000'
line_color = 'darkgreen'
# цвет линий функции, можно в формате HEX '#FF0000'
f_color = 'orangered'


########################################################################################################################
########################################################################################################################
def f(x):
    x1 = x[0]
    x2 = x[1]
    return k1 * x1 ** 2 + k2 * x1 * x2 + k3 * x2 ** 2 + k4 * x1 + k5 * x2


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

    for i in range(3):
        x = trajectory[-1] - np.dot(f_A, grad(trajectory[-1]))
        print(f'{1 + i}.J\' = ', grad(trajectory[-1]))
        trajectory.append(x)

    resx, resy = [], []
    print("Points: ")
    for t in trajectory:
        print(t)
        resx.append(t[0])
        resy.append(t[1])
    return resx, resy


def main():
    plt.figure(figsize=(10, 8))
    plot_equations(koef)
    resx, resy = one(start_point)
    plt.plot(resx, resy, color=line_color)
    plt.savefig("one.png")
    plt.show()


if __name__ == '__main__':
    main()
