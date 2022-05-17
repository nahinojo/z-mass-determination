import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
from scipy.optimize import curve_fit


## Defining relevant functions
def f_voigt(x, alpha, gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma/np.sqrt(2*np.pi) 

def f_fall(x, N, tau):
    return N * np.exp(-x / tau) 

def fit_func(x, alpha, gamma, N, tau, A, s):
    return A * ( ((1-s)*f_fall(x, N, tau)) + (s*f_voigt(x, alpha, gamma)) )


## Initial objects
imass_df = pd.read_csv(open("data\\invariant_mass.csv"))


## Plotting invariant mass as histogram
bins = 80
imass_hist = plt.hist(imass_df["Invariant Mass"],
                      bins = bins,
                      range = (70,110),
                      color = 'limegreen')
plt.title("Muon Invariant Mass")


## Finding parameters to fit curve
# Every value in x_vals represents the middle point of every bin.
x_vals = []
for i in range(len(imass_hist[1]) - 1):
    avg = (imass_hist[1][i] + imass_hist[1][i + 1])/2
    x_vals.append(avg)
y_vals = imass_hist[0]
fit_param = curve_fit(fit_func,x_vals,y_vals, maxfev = 50000)[0]

print(fit_param)

## Plotting fitted function to histogram
x_vals = np.linspace(70,110,bins)
plt.plot(x_vals, fit_func(x_vals,*fit_param), 'r')

# Saving histogram. 
plt.savefig("plots\\muon_hist.png")
