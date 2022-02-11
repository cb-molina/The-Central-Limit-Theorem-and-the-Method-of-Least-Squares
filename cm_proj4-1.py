'''
    Christian B. Molina
    Phy 104B - Project 04 Central Limit Theorem
    WQ 2021

    Box 6-1
'''
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Initialization
x = np.array([0.25, 1.05, 2.25, 2.88, 2.97, 3.64, 3.92, 4.94, 5.92])
y = np.array([0.86, 2.18, 4.84, 5.80, 6.99, 8.84, 8.71, 11.98, 12.40])
sigma = np.array([0.27, 1.16, 1.14, 0.93, 0.31, 0.66, 0.98, 0.93, 0.60])
omega = []
for i in sigma: omega.append(1) if i == 0 else omega.append(1/i**2)

# Calculations
alpha, delta = (np.sum(1 / sigma**2), np.sum((x / sigma)**2))
gamma = beta = np.sum(x / sigma**2)
theta, phi = (np.sum(y / sigma**2), np.sum(x*y / sigma**2))

det_D = alpha*delta - beta**2

a, b =  ((theta*delta - beta*phi) / det_D, (alpha*phi - theta*gamma) / det_D)
sigma_a, sigma_b, sigma_ab = ((delta / det_D)**(1/2), (alpha / det_D)**(1/2), -beta / det_D)



# Outputs
print("a = {:.4f} sigma_a = {:.4f}" .format(a,sigma_a))
print("b = {:.4f} sigma_b = {:.4f}" .format(b,sigma_b))
print("sigma_ab = {:.4f} " .format(sigma_ab))

plt.figure(figsize=(8,6))
plt.title("Linear Least-Squares Fit")
plt.errorbar(x,y, yerr = sigma, fmt = '.', label = "Data from Table 6-2 of Wong")
plt.plot(x, a + b*x,'--', label = "Linear Best-fit: y = a + b*x\nwhere a = {:.2f} and b = {:.2f}".format(a,b))
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.show()