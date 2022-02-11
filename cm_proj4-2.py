'''
    Christian B. Molina
    Phy 104B - Project 04 Central Limit Theorem
    WQ 2021

    Problem 6-3:
        Use a random number generator with an even distribution in the range [-1,+1] to produce n=6 values and store the sum as x. 
        Collect 1000 such sums and plot their distribution. Compare the results with a normal distribution of the same mean and variance
        as the x collected. Calculate the X^2-value. Repeat the calculations with n=50. Compare the two X^2-values obtained.
'''
import numpy as np
import random as r
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chisquare
plt.style.use('seaborn-darkgrid')

# Initialization
r.seed(None)

def distribution(n):
    # Function Init
    n_bins = 25
    x, sigma = ([],[])
    # Generate Data
    for i in range(1000): # Collecting 1000 sums
        j, rand_sum = (0,0)
        while j < n:
            rand_sum = rand_sum + r.uniform(-1,1)
            j += 1
        x.append(rand_sum)

    # Finding Average and Standard Deviation
    mu, sigma = norm.fit(x)

    # Plot histogram and calculate chi-squared
    plt.figure(figsize=(8,6))
    hist_data = plt.hist(x,n_bins,density = True,edgecolor='black',alpha=0.5)
    chisq = chisquare(hist_data[0])

    # Generate Normal Distribution
    xmin, xmax = plt.xlim()
    xval = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(xval, mu, sigma)
    plt.plot(xval,pdf,label="Normal Distribution\nwhere $\sigma$ = {:.2f} and $\mu$ = {:.2f}".format(sigma,mu))

    plt.title("Comparison of Normal Distribution over Histogram\nwhere n = {}".format(n))
    plt.xlabel('Value of the sum of 6 numbers'); plt.ylabel('Number of Occurrences')
    plt.plot([],[],' ',label="$\chi^2$ = {:.2f}".format(chisq[0]))
    plt.legend()


distribution(6)
distribution(50)
plt.show()