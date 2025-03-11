import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, simpson


def f(x,p1,p2,p3,p4):
    return p1 + p2*x + p3*x**2 + p4*x**3

def integration(p1,p2,p3,p4,a,b) :
    return (p1 * b + (p2 / 2) * b ** 2 + (p3 / 3) * b ** 3 + (p4 / 4) * b ** 4) - (p1 * a + (p2 / 2) * a ** 2 + (p3 / 3) * a ** 3 + (p4 / 4) * a ** 4)

def methode_rectangle_numpy(p1,p2,p3,p4,n,a,b):
    x = np.linspace(a, b, n)  # Points d'évaluation
    y = f(x, p1, p2, p3, p4)  # Valeurs de la fonction

    dx = (b - a) / (n - 1)  # Largeur constante des rectangles
    x_rect = x[:-1]  # Prendre x sauf le dernier point pour les bases des rectangles
    y_rect = y[:-1]  # Hauteur des rectangles (valeur gauche)

    # Calcul de l'intégrale sans boucle
    integrale = np.sum(y_rect * dx)

    # Tracé de la courbe
    #plt.plot(x, y, "bo-", label="f(x)")

    # Construction des coordonnées des rectangles sans boucle
    X_rect = np.array([x_rect, x_rect, x_rect + dx, x_rect + dx, x_rect]).T
    Y_rect = np.array([np.zeros_like(y_rect), y_rect, y_rect, np.zeros_like(y_rect), np.zeros_like(y_rect)]).T

    # Tracé des rectangles
    #plt.plot(X_rect.T, Y_rect.T, "r")

    #plt.legend(["f(x)", "Rectangles"])
    #plt.show()
    #plt.show()
    return integrale

def methode_rectangle_python(p1, p2, p3, p4,n,a,b):
    ni = n - 1  # nombre d'intervalles
    x_values = [a + i * (b - a) / ni for i in range(n)]
    y_values = [f(x, p1, p2, p3, p4) for x in x_values]

    #plt.plot(x_values, y_values, "bo-")

    integrale = 0
    for i in range(ni):
        dx = x_values[i+1] - x_values[i]
        integrale += y_values[i] * dx

        x_rect = [x_values[i], x_values[i], x_values[i+1], x_values[i+1], x_values[i]]
        y_rect = [0, y_values[i], y_values[i], 0, 0]
        #plt.plot(x_rect, y_rect, "r")
    #plt.show()
    return integrale

def methode_trapeze_python(p1,p2,p3,p4,n,a,b):
    dx = (b-a)/n
    somme =  0.5 * (f(a, p1, p2, p3, p4) + f(b, p1, p2, p3, p4))
    for i in range(1, n):
        x_i = a + i * dx
        somme = somme + f(x_i, p1, p2, p3, p4)
    return somme * dx

def methode_trapeze_numpy(p1,p2,p3,p4,n,a,b):
    dx = (b-a)/n
    x = np.linspace(a,b,n)
    y = f(x,p1, p2, p3, p4)
    return (dx/2) * np.sum(y[:-1]+ y[1:])

def methode_simpson_python(p1,p2,p3,p4,n,a,b):
    dx = (b - a) / n
    somme = (f(a,p1, p2, p3, p4) + f(b,p1, p2, p3, p4)) / 2 + 2 * f((a + dx / 2),p1, p2, p3, p4)
    for i in range(1, n):
        x_i = a + i * dx
        somme = somme + f(x_i,p1, p2, p3, p4) + 2 * f((x_i + dx / 2),p1, p2, p3, p4)

    return somme * dx / 3

def methode_simpson_numpy(p1,p2,p3,p4,n,a,b):
    dx = (b-a) / n
    x = np.linspace(a,b,n)
    y = f(x,p1, p2, p3, p4)
    return (dx / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

def methode_trapezes_scipy(p1,p2,p3,p4,n,a,b):
    x = np.linspace(a, b, n + 1)
    y = f(x, p1, p2, p3, p4)
    return trapezoid(y, x)

def methode_simpson_scipy(p1,p2,p3,p4,n,a,b):
    x = np.linspace(a, b, n + 1)
    y = f(x, p1, p2, p3, p4)
    return simpson(y, x)


def erreur(integrale):
    return 100*abs((integrale - integration(1, 2, 3, 4, 0, 1))/integration(1, 2, 3, 4, 0, 1))



