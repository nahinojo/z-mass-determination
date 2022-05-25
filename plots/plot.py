# Importing math-related libraries.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
from scipy.special import voigt_profile as f_voigt
from scipy.optimize import curve_fit


## Plotting invariant mass on histogram.
# Invariant mass data.
imass_df = pd.read_csv(open("data\\invariant_mass.csv"))
# Histogram parameters.
bins = 80
# Plotting histogram.
imass_hist = plt.hist(imass_df["Invariant Mass"],
                      bins = bins,
                      range = (70,110),
                      color = "firebrick",
                      label = "Muon IM")
plt.title("Invariant Mass")
plt.xlabel("Mass [GeV]")
plt.ylabel("Entities / bin")


## Defining function for curve-fitting.
def f_fall(x, xi_fall, N, tau): 
    return N*np.exp(-(x-xi_fall)/tau)

def f_fit(x, A, s, 
          xi_voigt=0, alpha=None, gamma=None,
          xi_fall=0, N=None, tau=None):
    sigma = alpha / np.sqrt(2*np.log(2))
    return A*(  (1 - s)*f_fall(x, xi_fall, N, tau) 
              + s*f_voigt(x - xi_voigt, sigma, gamma)  )


## Finding parameters to fit curve.
# Every x value represents the middle point of each histogram bin.
x_vals = []
for i in range(len(imass_hist[1]) - 1):
    avg = (imass_hist[1][i] + imass_hist[1][i + 1])/2
    x_vals.append(avg)
# Every y value is simple the size of the size/height of the histogram bin. 
y_vals = imass_hist[0]
# Calculating parameters using curve_fit().
param = curve_fit(f_fit,                            
                  x_vals,                          
                  y_vals,                             
                  p0=[1, 1, 91, 1, 1, 69, 1, 1], # Guessing the x-shift of each function as 91 and 69.
                  maxfev=3000)[0]                 
# Printing calculated parameters.
print(100*"*")
print("Calculated Parameters:")
print("[A, s, xi_voight, alpha,")
print("gamma, xi_fall, N, tau]")
print("")
print(param)
print(100*"*")


## Plotting fitted function to histogram.
# Alterting list of x values to be a numpy array for plotting.
x_vals = np.asarray(x_vals)
# Plotting the function given the x values and parameters. 
plt.plot(x_vals, f_fit(x_vals,*param),
         "black", linestyle='dashed',label="Sig.+Bg.")
plt.plot(x_vals, param[1]*(1-param[2])*f_fall(x_vals,*param[-3:]),
         "lightsalmon", linestyle='dotted',label="Backg.")
plt.legend()
# Saving histogram. 
plt.savefig("plots\\muon_hist.png")

