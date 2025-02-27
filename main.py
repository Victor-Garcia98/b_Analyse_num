import numpy as np
import matplotlib

def f(x,p1,p2,p3,p4):
    return p1 + p2*x + p3*x**2 + p4*x**3

def integration(p1,p2,p3,p4,a,b) :
    return (p1 * b + (p2 / 2) * b ** 2 + (p3 / 3) * b ** 3 + (p4 / 4) * b ** 4) - (p1 * a + (p2 / 2) * a ** 2 + (p3 / 3) * a ** 3 + (p4 / 4) * a ** 4)




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




