from scipy.optimize import curve_fit
from scipy.special import voigt_profile as f_voigt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print(10 * "*")


"""
Setting up invariant mass Dataframe.

Filetering events where either muon is below the isolation upper boundary.
"""
muon_df = pd.read_csv(open("data\\mini_muons.csv"), index_col=0)
imass_df = pd.read_csv(open("data\\invariant_mass.csv"), index_col=0)
iso_upper_bound = .084
imass_df[["Iso 1", "Iso 2"]] = muon_df[["Iso 1", "Iso 2"]]
imass_df = pd.merge(
    imass_df[imass_df["Iso 1"] < iso_upper_bound],
    imass_df[imass_df["Iso 2"] < iso_upper_bound],
    how='inner',
    on=list(imass_df.columns)
)


"""
Histogram generation
"""
bins = 60
rng = (70, 110)
imass_hist = plt.hist(
    imass_df["Invariant Mass"],
    bins=bins,
    range=rng,
    color="firebrick",
    label="Muon IM"
)
# Extracting y-ticks of original, non-normalized histogram.
ytick_original = [int(y) for y in plt.yticks()[0]]
plt.clf()

# Recreating new, normalized histogram
imass_hist = plt.hist(
    imass_df["Invariant Mass"],
    bins=bins,
    range=rng,
    density=True,
    color="firebrick",
    label="OS Muons"
)
plt.yticks(np.linspace(0, plt.yticks()[0][-1], num=len(ytick_original)), ytick_original)
plt.title("Muons Invariant Mass — Isolation <" + str(iso_upper_bound))
plt.title("Invariant Mass")
plt.xlabel("Mass [GeV]")
plt.ylabel("Entities / bin")


"""
Defining functions and inputs for curve-fitting.
"""
x_vals = []
for i in range(len(imass_hist[1]) - 1):
    avg = (imass_hist[1][i] + imass_hist[1][i + 1]) / 2
    x_vals.append(avg)
y_vals = imass_hist[0]
def f_fall(x, xi_fall, tau):
    def E(x, xi_fall, tau): return np.exp(-(x - xi_fall) / tau)
    return (E(x, xi_fall, tau)
            / np.trapz(E(x_vals, xi_fall, tau), x_vals))
# Parent fucntion.
def f_fit(x, s, xi_voigt=0, xi_fall=0,
          alpha=None, gamma=None, tau=None):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return ((1 - s) * f_fall(x, xi_fall, tau)
            + s * f_voigt(x - xi_voigt, sigma, gamma))


"""
Curve-fitting f_fit to histogram bin heights.
"""
para_bounds = np.vstack((np.zeros(6), np.inf * np.ones(6)))
para_bounds[:, 0] = (.25, 1)  # s boundaries
para, para_var = curve_fit(
    f_fit,
    x_vals,
    y_vals,
    p0=[.7, 91, 50, 2, 1, 1],
    bounds=para_bounds,
    maxfev=10000
)
signal_fraction = para[0]
signal_uncertain = np.sqrt(para_var[0, 0])


print("Calculated Parameters:")
print("[s, xi_voigt, xi_fall, ")
print(" alpha, gamma, tau]")
print("")
print(para)
print("")
print("")
print("Paramater Variance Matrix:")
print(para_var)
print("")


"""
Plotting Sig. Backg. and Background only functions to histogram.
"""
x_vals = np.asarray(x_vals)
plt.plot(
    x_vals, 
    f_fit(x_vals, *para),
    color="black", 
    linestyle='dashed', 
    label="Sig.+Backg."
)
plt.plot(
    x_vals, 
    (1 - para[0]) * f_fall(x_vals, para[2], para[5]),
    color="lightsalmon",
    linestyle='dotted',
    label="Background"
)

"""
Adding signal fraction & uncertainty to legend.

The conditional for signal fraction prevents the number from rouning
to 1.0. Instead, s caps at .999
"""
if signal_fraction <= .9994:
    signal_fraction_string = "{:.3f}".format(signal_fraction)[1:]
else:
    signal_fraction_string = ".999"
plt.legend(title="Sig. Frac. = "
           + signal_fraction_string
           + "±"
           + "{:.3f}".format(signal_uncertain)[1:])
plt.savefig("plots\\imass_hist.png")


print("Signal Fraction:", signal_fraction)
print("SF Uncertainty:", signal_uncertain)

