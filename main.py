import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, simpson
import timeit


def f(x, p1, p2, p3, p4):
    """
    Fonction définissant un polynôme du troisième degré :
    f(x) = p1 + p2*x + p3*x^2 + p4*x^3
    """
    return p1 + p2*x + p3*x**2 + p4*x**3

def integration(p1, p2, p3, p4, a, b):
    """
    Calcul de l'intégrale analytique du polynôme sur l'intervalle [a, b]
    Utilise les primitives des termes individuels :
    ∫(p1 + p2*x + p3*x^2 + p4*x^3) dx
    """
    return (p1 * b + (p2 / 2) * b ** 2 + (p3 / 3) * b ** 3 + (p4 / 4) * b ** 4) - \
           (p1 * a + (p2 / 2) * a ** 2 + (p3 / 3) * a ** 3 + (p4 / 4) * a ** 4)

def methode_rectangle_numpy(p1, p2, p3, p4, n, a, b):
    """
    Intégration numérique par la méthode des rectangles avec NumPy.
    - On découpe l'intervalle [a, b] en n sous-intervalles.
    - L'aire est approximée en prenant la valeur de f(x) à gauche de chaque rectangle.
    """
    x = np.linspace(a, b, int(n))  # Points d'évaluation
    y = f(x, p1, p2, p3, p4)  # Valeurs de la fonction

    dx = (b - a) / (n - 1)  # Largeur des rectangles
    x_rect = x[:-1]  # Coordonnées des bases des rectangles
    y_rect = y[:-1]  # Hauteur des rectangles

    integrale = np.sum(y_rect * dx)  # Somme des aires des rectangles
    return integrale

def methode_rectangle_python(p1, p2, p3, p4, n, a, b):
    """
    Intégration par la méthode des rectangles en Python pur.
    - Utilise une boucle for au lieu de vectorisation NumPy.
    """
    ni = n - 1  # Nombre de sous-intervalles
    x_values = [a + i * (b - a) / ni for i in range(int(n))]  # Points d'évaluation
    y_values = [f(x, p1, p2, p3, p4) for x in x_values]  # Valeurs de la fonction

    integrale = 0
    for i in range(int(ni)):  # Sommation des aires des rectangles
        dx = x_values[i+1] - x_values[i]  # Largeur
        integrale += y_values[i] * dx  # Hauteur * Largeur

    return integrale

def methode_trapeze_python(p1, p2, p3, p4, n, a, b):
    """
    Intégration numérique par la méthode des trapèzes en Python pur.
    - Approximations par des trapèzes au lieu de rectangles.
    """
    dx = (b - a) / n  # Largeur des sous-intervalles
    somme = 0.5 * (f(a, p1, p2, p3, p4) + f(b, p1, p2, p3, p4))  # Moyenne des bornes

    for i in range(1, int(n)):  # Sommation des hauteurs intermédiaires
        x_i = a + i * dx
        somme += f(x_i, p1, p2, p3, p4)

    return somme * dx

def methode_trapeze_numpy(p1, p2, p3, p4, n, a, b):
    """
    Intégration par la méthode des trapèzes avec NumPy.
    - Utilisation de linspace et sum() pour vectoriser les calculs.
    """
    dx = (b - a) / n
    x = np.linspace(a, b, int(n))  # Points d'évaluation
    y = f(x, p1, p2, p3, p4)  # Valeurs de la fonction

    return (dx / 2) * np.sum(y[:-1] + y[1:])  # Moyenne des points adjacents

def methode_simpson_python(p1, p2, p3, p4, n, a, b):
    """
    Intégration numérique par la méthode de Simpson en Python pur.
    - Approximation par une parabole passant par trois points.
    """
    dx = (b - a) / n
    somme = (f(a, p1, p2, p3, p4) + f(b, p1, p2, p3, p4)) / 2 + 2 * f((a + dx / 2), p1, p2, p3, p4)

    for i in range(1, int(n)):
        x_i = a + i * dx
        somme += f(x_i, p1, p2, p3, p4) + 2 * f((x_i + dx / 2), p1, p2, p3, p4)

    return somme * dx / 3

def methode_simpson_numpy(p1,p2,p3,p4,n,a,b):
    dx = (b-a) / n
    x = np.linspace(a,b,int(n))
    y = f(x,p1, p2, p3, p4)
    return (dx / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

def methode_trapeze_scipy(p1,p2,p3,p4,n,a,b):
    x = np.linspace(a, b, int(n + 1))
    y = f(x, p1, p2, p3, p4)
    return trapezoid(y, x)

def methode_simpson_scipy(p1,p2,p3,p4,n,a,b):
    x = np.linspace(a, b, int(n + 1))
    y = f(x, p1, p2, p3, p4)
    return simpson(y, x)


def erreur(integrale, p1, p2, p3, p4, n, a, b):
    """
    Calcul de l'erreur entre l'intégrale exacte et celle calculée numériquement.
    """
    integrale_exact = integration(p1, p2, p3, p4, a, b)
    integrale_methode = integrale(p1, p2, p3, p4, n, a, b)
    return abs(integrale_exact - integrale_methode)


def affichage_convergence_total(p1,p2,p3,p4,n_max,a,b):

    n_val = np.linspace(10,n_max,100)
    erreur_dict = {'rectangles_python': [],'rectangles_numpy': [],'trapèzes_python': [],'trapèzes_numpy': []
        ,'trapèzes_scipy': [],'simpson_python': [],'simpson_numpy': [],'simpson_scipy': []}

    for n in n_val:
        erreur_dict['rectangles_python'].append(erreur(methode_rectangle_python,p1,p2,p3,p4,n,a,b))
        erreur_dict['rectangles_numpy'].append(erreur(methode_rectangle_numpy,p1,p2,p3,p4,n,a,b))
        erreur_dict['trapèzes_python'].append(erreur(methode_trapeze_python,p1,p2,p3,p4,n,a,b))
        erreur_dict['trapèzes_numpy'].append(erreur(methode_trapeze_numpy,p1, p2, p3, p4, n, a, b))
        erreur_dict['trapèzes_scipy'].append(erreur(methode_trapeze_scipy, p1, p2, p3, p4, n, a, b))
        erreur_dict['simpson_python'].append(erreur(methode_simpson_python, p1, p2, p3, p4, n, a, b))
        erreur_dict['simpson_numpy'].append(erreur(methode_simpson_numpy, p1, p2, p3, p4, n, a, b))
        erreur_dict['simpson_scipy'].append(erreur(methode_simpson_scipy, p1, p2, p3, p4, n, a, b))

    plt.figure()
    for methode, error in erreur_dict.items():
        plt.plot(n_val, error, label=f"Méthode des {methode}", marker='+')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Nombre de segments n')
    plt.ylabel('Erreur par rapport au numérique')
    plt.title('Convergence des méthodes d\'intégration')
    plt.legend()
    plt.show()

def mesure_temps_execution(methode, p1, p2, p3, p4, n, a, b):
    """
    Mesure du temps d'exécution d'une méthode donnée.
    """
    return timeit.timeit(lambda: methode(p1, p2, p3, p4, n, a, b), number=1)

def affichage_temps_total(p1,p2,p3,p4,n_max,a,b):
    n_val = np.linspace(10,n_max,100)
    temps_dict = {'rectangles_python': [],'rectangles_numpy': [],'trapèzes_python': [],'trapèzes_numpy': []
        ,'trapèzes_scipy': [],'simpson_python': [],'simpson_numpy': [],'simpson_scipy': []}

    for n in n_val:
        temps_dict['rectangles_python'].append(mesure_temps_execution(methode_rectangle_python,p1,p2,p3,p4,n,a,b))
        temps_dict['rectangles_numpy'].append(mesure_temps_execution(methode_rectangle_numpy,p1,p2,p3,p4,n,a,b))
        temps_dict['trapèzes_python'].append(mesure_temps_execution(methode_trapeze_python,p1,p2,p3,p4,n,a,b))
        temps_dict['trapèzes_numpy'].append(mesure_temps_execution(methode_trapeze_numpy,p1, p2, p3, p4, n, a, b))
        temps_dict['trapèzes_scipy'].append(mesure_temps_execution(methode_trapeze_scipy, p1, p2, p3, p4, n, a, b))
        temps_dict['simpson_python'].append(mesure_temps_execution(methode_simpson_python, p1, p2, p3, p4, n, a, b))
        temps_dict['simpson_numpy'].append(mesure_temps_execution(methode_simpson_numpy, p1, p2, p3, p4, n, a, b))
        temps_dict['simpson_scipy'].append(mesure_temps_execution(methode_simpson_scipy, p1, p2, p3, p4, n, a, b))

    plt.figure()
    for methode, temps in temps_dict.items():
        plt.plot(n_val, temps, label=f"Méthode des {methode}", marker='+')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Nombre de segments n')
    plt.ylabel('temps d\'éxécution')
    plt.title('temps d\'éxécution des méthodes d\'intégration')
    plt.legend()
    plt.show()

def demande_utilisateur():
    print('Pour l\'étude comparative, nous allons vous demander dans l\'ordre les différents coefficient du polynome puis '
          'le nombre de segments maximums envisagés et pour finir les bornes d\'intégration \n\n'
          'le polynome est de la forme : P1 + P2 x + P3 x^2 + P4 x^3 = y')
    p1 = int(input('p1 ='))
    p2 = int(input('p2 ='))
    p3 = int(input('p3 ='))
    p4 = int(input('p4 ='))
    n_max = int(input ('nombre de segemnts maximums = '))
    a = int(input ('borne d\'intégration inférieure = '))
    b = int(input('borne d\'intégration supérieure = '))
    return p1,p2,p3,p4,n_max,a,b


if __name__ == "__main__":
    [p1,p2,p3,p4,n_max,a,b] = demande_utilisateur()
    affichage_convergence_total(p1, p2, p3, p4, n_max, a, b)
    affichage_temps_total(p1, p2, p3, p4, n_max, a, b)






