import torch
import f1
import math


def h(x):
    return math.sqrt((math.sin(x) ** 2) + 1)


def h_prim(x):
    return (0.5 * math.sin(2 * x)) / math.sqrt((math.sin(x) ** 2) + 1)


def h_prim_prim(x):
    
    return -0.25 * ((1+(math.sin(x)**2))**-1.5) * ((math.sin(2*x)) ** 2)  + (math.cos(2*x)) * ((1+ (math.sin(x)**2) )**-0.5)
   


def f2(x,A=None):
    value = h(f1.phy(x))
    return value
def f2grad(x,A=None):
    grad = h_prim(f1.phy(x)) * f1.phygrad(x)
    return grad
def f2hessian(x,A=None):
    hess = h_prim(f1.phy(x)) * f1.phyhessian(x) + h_prim_prim(f1.phy(x)) * (f1.phygrad(x) @ f1.phygrad(x).T)
    return hess