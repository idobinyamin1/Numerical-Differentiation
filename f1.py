import torch
import math


# 1.15
# 1
def f1(x: torch.Tensor, A: torch.Tensor):

    value = phy(A @ x)   
    return value

def f1grad(x: torch.Tensor, A: torch.Tensor):
    gradient = A.T @ phygrad(A @ x)
    return gradient
def f1hessian(x: torch.Tensor, A: torch.Tensor):
    hessian = A.T @ phyhessian(A @ x) @ A
    return hessian

def u(x: torch.tensor) -> float:
    
    y = x
    y.reshape(3, )
    return (y[0]) * (y[1] ** 2) * (y[2])


def ugrad(x: torch.tensor) -> torch.tensor:
   

    y = x
    y.reshape(3, )
    return torch.tensor([((y[1] ** 2) * y[2]), (2 * y[0] * y[1] * y[2]), (y[0] * (y[1] ** 2))]).reshape(3, 1)


def uhassian(x: torch.tensor) -> torch.tensor:
    
    y = x
    y.reshape(3, )
    res=torch.tensor([[0, 2 * y[1] * y[2], y[1] ** 2],
                         [2 * y[1] * y[2], 2 * x[0] * y[2], 2*y[0] * y[1]],
                         [y[1] ** 2, 2 * y[0] * y[1], 0]])
    assert res.shape == (3, 3), 'size mismatch'
    return res


def phy(x: torch.tensor) -> float:
 
    return math.cos((u(x))) ** 2


def phygrad(x: torch.tensor) -> torch.tensor:
   
    return -ugrad(x) * math.sin(2 * u(x))


def phyhessian(x: torch.tensor) -> torch.tensor:

    t1=- math.sin(2 * u(x)) * uhassian(x)
   
    t2=- 2 * math.cos(2 * u(x)) * ugrad(x) @ ugrad(x).T
      
    return t1+t2

