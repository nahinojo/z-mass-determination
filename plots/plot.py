import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
from scipy.special import voigt_profile
from scipy.optimize import curve_fit

## Constraints for curve fitting.
# Shifting x of both functions to center near histogram peak.
x_shift_voigt = 90.7
x_shift_fall = 68
# Graph colors
hist_clr = "firebrick"
func_clr = "black"
func_fall_clr = "lightsalmon"

## Defining relevant functions
def f_voigt(x, alpha, gamma):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma/np.sqrt(2*np.pi) 

def f_fall(x, N, tau):
    return N * np.exp(-x / tau) 

def func(x, alpha, gamma, N, tau, A, s):
    return A*( ((1 - s)*f_fall(x - x_shift_fall, N, tau)) 
              + (s*f_voigt(x - x_shift_voigt, alpha, gamma)) )

def func_fall(x, N, tau, A, s):
    return A*((1 - s)*f_fall(x - x_shift_fall, N, tau))

def func_voigt(x, alpha, gamma, A, s):
    return A*(s*f_voigt(x - x_shift_voigt, alpha, gamma))



## Initial objects
imass_df = pd.read_csv(open("data\\invariant_mass.csv"))


## Plotting invariant mass as histogram.
bins = 80
imass_hist = plt.hist(imass_df["Invariant Mass"],
                      bins = bins,
                      range = (70,110),
                      color = hist_clr,
                      label = "Muon IM")
plt.title("Muon Invariant Mass")
plt.xlabel("Mass [GeV]")
plt.ylabel("Entities / bin")


## Finding parameters to fit curve
# Every x value represents the middle point of each histogram bin.
x_vals = []
for i in range(len(imass_hist[1]) - 1):
    avg = (imass_hist[1][i] + imass_hist[1][i + 1])/2
    x_vals.append(avg)
# Every y value is simple the size of the size/height of the histogram bin. 
y_vals = imass_hist[0]
# Calculates best parameters using curve_fit().
fit_param = curve_fit(func, 
                      x_vals,
                      y_vals,
                      maxfev = 3000)[0] # [alpha, gamma, N, tau, A, s]
print(100*"*")
print("Calculated Parameters:")
print("")
print(fit_param)
print(100*"*")


## Plotting fitted function to histogram
# Alterting list of x values to be a numpy array for plotting
x_vals = np.asarray(x_vals)
# Plotting the function, given the x values and parameters. 
plt.plot(x_vals, func(x_vals,*fit_param), func_clr, linestyle='dashed',label="Sig.+Bg.")
plt.plot(x_vals,func_fall(x_vals,*fit_param[2:]), func_fall_clr, linestyle='dotted',label="Backg.")
plt.legend()
# Saving histogram. 
plt.savefig("plots\\muon_hist.png")
