import torch
import f1
import f2
import numerical_calculations
import matplotlib.pyplot as plt
import math


#input for test
x = torch.rand(3,1, dtype=torch.float64)
A = torch.rand(3,3, dtype=torch.float64)


#inf norm
def calc_inf_norm(vec):
    return torch.max(torch.abs(vec)).item()

#

#initializing lists for graphs
inf_norm_grad1 = []
inf_norm_grad2 = []
inf_norm_hess1 = []
inf_norm_hess2 = []

rng=range(0, 60)
#main loop for testing
for i in rng:
    epsilon = 2 ** (-1 * i)
    
    #f1 calculation
   
    grad1=f1.f1grad(x, A)
    hess1=f1.f1hessian(x, A)

    numerical_grad1 = numerical_calculations.calc_grad_f(f1.f1, x, epsilon,A)
    numerical_hess1 = numerical_calculations.calc_hess_f(f1.f1grad, x, epsilon,A)


    #f2 calculation
    
    grad2=f2.f2grad(x)
    hess2=f2.f2hessian(x)
    numerical_grad2 = numerical_calculations.calc_grad_f(f2.f2, x, epsilon)
    numerical_hess2 = numerical_calculations.calc_hess_f(f2.f2grad, x, epsilon)


    #comparion between analytical and numirical calculations
    inf_norm_grad1.append(calc_inf_norm(grad1 - numerical_grad1))
    inf_norm_grad2.append(calc_inf_norm(grad2 - numerical_grad2))
    inf_norm_hess1.append(calc_inf_norm(hess1 - numerical_hess1))
    inf_norm_hess2.append(calc_inf_norm(hess2 - numerical_hess2))
# bulding graphs
figure, axis = plt.subplots(2, 2)

axis[0, 0].set_xlabel("$log_{2} epsilon$")
axis[0, 0].set_ylabel("$||error||_{inf}$")
axis[0, 0].set_title("f1 gradient")
axis[0, 0].plot(rng, inf_norm_grad1)


axis[0, 1].set_xlabel("$log_{2} epsilon$")
axis[0, 1].set_ylabel("$||error||_{inf}$")
axis[0, 1].set_title("f2 gradient")
axis[0, 1].plot(rng, inf_norm_grad2)


axis[1, 0].set_xlabel("$log_{2} epsilon$")
axis[1, 0].set_ylabel("$||error||_{inf}$")
axis[1, 0].set_title("f1 hessian ")
axis[1, 0].plot(rng, inf_norm_hess1)

axis[1, 1].set_xlabel("$log_{2} epsilon$")
axis[1, 1].set_ylabel("$||error||_{inf}$")
axis[1, 1].set_title("f2 hessian")
axis[1, 1].plot(rng, inf_norm_hess2)
plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9   , top=0.9, wspace=0.7, hspace=0.7)

# Combine all the operations and display
print("best epsilon for f1 gradient","2^-",torch.argmin(torch.tensor(inf_norm_grad1)).item())
print("best epsilon for f2 hessian","2^-",torch.argmin(torch.tensor(inf_norm_grad2)).item())

print("best epsilon for f1 gradient","2^-",torch.argmin(torch.tensor(inf_norm_hess1)).item())
print("best epsilon for f2 hessian","2^-",torch.argmin(torch.tensor(inf_norm_hess2)).item())


plt.show()

