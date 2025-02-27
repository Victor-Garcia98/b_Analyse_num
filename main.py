import numpy as np
import matplotlib.pyplot as plt

def f(x,p1,p2,p3,p4):
    return p1 + p2*x + p3*x**2 + p4*x**3

def integration(p1,p2,p3,p4,a,b) :
    return (p1 * b + (p2 / 2) * b ** 2 + (p3 / 3) * b ** 3 + (p4 / 4) * b ** 4) - (p1 * a + (p2 / 2) * a ** 2 + (p3 / 3) * a ** 3 + (p4 / 4) * a ** 4)


integration_numerique = integration(1, 2, 3, 4, 0, 1)
print(integration_numerique)


def methode_rectangle_numpy(n,xmin,xmax,p1,p2,p3,p4):
    ni = n - 1 # nombre d'intervalles

    x = np.linspace(xmin, xmax, n)
    y = f(x,p1,p2,p3,p4)
    plt.plot(x,y,"bo-")

    integrale = 0
    for i in range(ni):
        integrale = integrale + y[i]*(x[i+1]-x[i])
        x_rect = [x[i], x[i], x[i+1], x[i+1], x[i]]
        y_rect = [0   , y[i], y[i]  , 0     , 0   ]
        plt.plot(x_rect, y_rect,"r")
    print("integrale =", integrale)

    plt.show()

print(methode_rectangle_numpy(50,0,1,1,2,3,4))

def methode_rectangle_python(n, xmin, xmax, p1, p2, p3, p4):
    ni = n - 1  # nombre d'intervalles
    x_values = [xmin + i * (xmax - xmin) / ni for i in range(n)]
    y_values = [f(x, p1, p2, p3, p4) for x in x_values]

    plt.plot(x_values, y_values, "bo-")

    integrale = 0
    for i in range(ni):
        dx = x_values[i+1] - x_values[i]
        integrale += y_values[i] * dx

        x_rect = [x_values[i], x_values[i], x_values[i+1], x_values[i+1], x_values[i]]
        y_rect = [0, y_values[i], y_values[i], 0, 0]
        plt.plot(x_rect, y_rect, "r")

    print("Intégrale =", integrale)
    plt.show()


print(methode_rectangle_python(50,0,1,1,2,3,4))

