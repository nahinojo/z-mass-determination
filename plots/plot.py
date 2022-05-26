print(100*"*")
# Importing math-related libraries.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import voigt_profile as f_voigt
from scipy.optimize import curve_fit


## Invariant mass dataframe. 
# Only keeping values within specific isolation boundaries.
muon_df = pd.read_csv(open("data\\mini_muons.csv"), index_col=0)
imass_df = pd.read_csv(open("data\\invariant_mass.csv"), index_col=0)
iso_bounds = (0, 1)
imass_df[["Iso 1","Iso 2"]] = muon_df[["Iso 1","Iso 2"]]
imass_df = pd.merge(imass_df[(imass_df["Iso 1"] > iso_bounds[0])
                             & (imass_df["Iso 1"] < iso_bounds[1])],
                    imass_df[(imass_df["Iso 2"] > iso_bounds[0])
                             & (imass_df["Iso 2"] < iso_bounds[1])],
                    how='inner',
                    on=list(imass_df.columns))
print("Inv. Mass Dataframe with bounded Isolation values",iso_bounds)
print("")
print(imass_df)
print(100*"-")


## Histogram generation.
# Defining histogram parameters.
bins = 80
rng = (70, 110)
# Plotting and design.
imass_hist = plt.hist(imass_df["Invariant Mass"],
                      bins=bins,
                      range=rng,
                      density=True, # Normalizes histogram
                      color="firebrick",
                      label="Muon IM")
plt.title("Inv. Mass, Normalized")
plt.xlabel("Mass [GeV]")
plt.ylabel("Entities / bin")


## x,y values for curve-fitting.
# x_vals represents the middle points of each histogram bin.
x_vals = []
for i in range(len(imass_hist[1]) - 1):
    avg = (imass_hist[1][i] + imass_hist[1][i + 1])/2
    x_vals.append(avg)
# Every y value is simple the size of the size/height of the histogram bin. 
y_vals = imass_hist[0]


## NORMALIZED functions for curve-fitting.
# Falling function.
def f_fall(x, xi_fall, tau):
    E = lambda x, xi_fall, tau: np.exp(-(x - xi_fall)/tau)
    return (E(x, xi_fall, tau) 
            / np.trapz(E(x_vals, xi_fall, tau), x_vals))
    
# Parent function.
def f_fit(x, s, xi_voigt=0, xi_fall=0,
          alpha=None, gamma=None, tau=None):
    sigma = alpha / np.sqrt(2*np.log(2))
    return ((1 - s)*f_fall(x, xi_fall, tau) 
            + s*f_voigt(x - xi_voigt, sigma, gamma))



## Bounding variables to be non-negative.
    # s should represent a ratio between V and E. 
    # alpha should be siginificantly large. 
para_bounds = np.vstack((np.zeros(6),np.inf*np.ones(6)))
para_bounds[:, 0] = (.25, 1) # s upper-bound
#param_bounds[0,2] =  # alpha lower-bound

print("Parameter Boundaries:")
print("[s, xi_fall, xi_voight,")
print("tau,  alpha, gamma]")
print("")
print(para_bounds)
print(100*"-")

# curve_fit() to calculate parameters.
[para, para_var] = curve_fit(f_fit,                            
                  x_vals,                          
                  y_vals,                             
                  p0=[.7, 91, 50, 2, 1, 1],
                  bounds=para_bounds,
                  maxfev=10000)                 

print("Calculated Parameters:")
print("[s, tau,  alpha, gamma")
print("xi_fall, xi_voight]")
print("")
print(para)
print("")
print("")
print("Paramater Variance Matrix:")
print(para_var)
print(100*"-")

## Plotting fitted function to histogram.
# Alterting list of x values to be a numpy array for plotting.
x_vals = np.asarray(x_vals)
# Plotting the function given the x values and parameters. 
plt.plot(x_vals, f_fit(x_vals,*para),
         "black", linestyle='dashed',label="Sig.+Bg.")
plt.plot(x_vals, (1-para[0])*f_fall(x_vals, para[2], para[5]),
         "lightsalmon", linestyle='dotted',label="Backg.")
plt.legend()
# Saving histogram. 
plt.savefig("plots\\muon_hist.png")


print("Signal Fraction:", para[0])
print("SF Uncertainty:", np.sqrt(para_var[0,0]))
print(100*"-")