import torch
import f1

e1 = torch.tensor([1, 0, 0], dtype=torch.float64).reshape(3, 1)
e2 = torch.tensor([0, 1, 0], dtype=torch.float64).reshape(3, 1)
e3 = torch.tensor([0, 0, 1], dtype=torch.float64).reshape(3, 1)


def calc_grad_f(f, x, epsilon,A=None):
    return torch.tensor([(f(x + epsilon * e1,A) - f(x - epsilon * e1,A)),
                         (f(x + epsilon * e2,A) - f(x - epsilon * e2,A)),
                         (f(x + epsilon * e3,A) - f(x - epsilon * e3,A))], dtype=torch.float64).reshape(3, 1) / (2 * epsilon)


def calc_hess_f(grad_f, x: torch.tensor(3, ), epsilon,A=None):
    
    g1=grad_f(x + epsilon * e1,A) - grad_f(x - epsilon * e1,A)

    g2=grad_f(x + epsilon * e2,A) - grad_f(x - epsilon * e2,A)
    g3=grad_f(x + epsilon * e3,A) - grad_f(x - epsilon * e3,A)
    
    
    res=torch.stack((g1,g2,g3),dim=1).reshape(3,3)
 
    return res/(2 * epsilon)


# eps = torch.pow(torch.tensor(2, dtype=torch.float64), -10)
# x = torch.tensor([1, 2, 3], dtype=torch.float64, requires_grad=True).reshape(3, 1)
# A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64, requires_grad=True)
#
#
# def calc_f1(vec):
#     v, g, h = f1.f1(vec, A)
#     return v


# v, g, h = f1.f1(x, A)

# print('omri')
# print(calc_grad_f(calc_f1, x, eps))
#
# print('ido')
# print(g)

# print('torch')
# v.backward(torch.FloatTensor([1.0, 1.0, 1.0]))
# torch.Tensor.backward()
# print((f1.phy(A@x)).backward)
# v.backward()
# v.retain_grad()
# print(v.grad)
