from turtle import color
from scipy.optimize import curve_fit
from scipy.special import voigt_profile as f_voigt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def fitted_histogram(
    iso_upper_bound: float = np.inf,
    bins: int = 60,
    mass_range: tuple = (70,110),
    hist_name: str = "imass_hist", 
    hist_only: bool = False,
    display_parameters: bool = False,
    display_signal: bool = False,
    return_signal_only: bool = False
):
    """
    Generates fitted histogram from given constraints.
    """ 
    
    # Filters invariant mass Dataframe given an isolation boundary.  
    imass_dataframe = pd.read_csv(open("data\\invariant_mass.csv"), index_col=0)
    muon_dataframe = pd.read_csv(open("data\\mini_muons.csv"), index_col=0)
    imass_dataframe[["Iso 1", "Iso 2"]] = muon_dataframe[["Iso 1", "Iso 2"]]
    imass_df = pd.merge(
        imass_dataframe[imass_dataframe["Iso 1"] < iso_upper_bound],
        imass_dataframe[imass_dataframe["Iso 2"] < iso_upper_bound],
        how='inner',
        on=list(imass_dataframe.columns)
    )
    
    # Generates normalized histogram with y-ticks of 
    # non-normalized counterpart.      
    imass_hist = plt.hist(
        imass_df["Invariant Mass"],
        bins=bins,
        range=mass_range,
        color="firebrick",
        label="Muon IM"
    )
    ytick_original = [int(y) for y in plt.yticks()[0]]
    plt.clf()
    
    imass_hist = plt.hist(
        imass_df["Invariant Mass"],
        bins=bins,
        range=mass_range,
        density=True,
        color="firebrick",
        label="OS Muons"
    )
    
    plt.xlabel("Mass [GeV]")
    plt.yticks(np.linspace(0, plt.yticks()[0][-1], num=len(ytick_original)), ytick_original)
    plt.ylabel("Entities / bin")
    if iso_upper_bound == np.inf:
        plt.title("Muons Invariant Mass — Any Isolation")
    else:
        plt.title("Muons Invariant Mass — Isolation <" + str(iso_upper_bound))


    # Unfitted histogram.
    if hist_only:
        plt.savefig("plots\\" + hist_name + ".png")
        return None
    
    # Retrieves x and y values of a histogram's bins. 
    x_vals = []
    for i in range(len(imass_hist[1]) - 1):
        avg = (imass_hist[1][i] + imass_hist[1][i + 1]) / 2
        x_vals.append(avg)
    y_vals = imass_hist[0]

    # Defining math functions for fitting
    f_fall_normfactor = lambda x, xi_fall, tau: (
        np.exp(-(x - xi_fall) / tau)
    )
    
    f_fall = lambda x, xi_fall, tau: (
        (f_fall_normfactor(x, xi_fall, tau)
        / np.trapz(f_fall_normfactor(x_vals, xi_fall, tau), x_vals))
    )
    
    f_fit = lambda x, s, xi_voigt, xi_fall, alpha, gamma, tau: (
        (1 - s) * f_fall(x, xi_fall, tau)
        + s * f_voigt(
            x - xi_voigt,
            alpha / np.sqrt(2 * math.log(2)), 
            gamma
        )
    )
    
    # Fitting using curve_fit() from SciPy.
    parameter_bounds = np.vstack((np.zeros(6), np.inf * np.ones(6)))
    parameter_bounds[:, 0] = (.25, 1)  # s boundaries
    parameters, parameter_variance = curve_fit(
        f_fit,
        x_vals,
        y_vals,
        p0=[.7, 91, 50, 2, 1, 1],
        bounds=parameter_bounds,
        maxfev=10000
    )
    
    if display_parameters:
        print("Calculated Parameters:")
        print("[s, xi_voigt, xi_fall, ")
        print(" alpha, gamma, tau]")
        print("")
        print(parameters)
        print("")
        print("")
        print("Paramater Variance Matrix:")
        print(parameter_variance)
        print("")
    
    x_vals = np.asarray(x_vals)
    plt.plot(
        x_vals, 
        f_fit(x_vals, *parameters),
        color="black", 
        linestyle='dashed', 
        label="Sig.+Backg."
    )
    
    plt.plot(
        x_vals, 
        (1 - parameters[0]) * f_fall(x_vals, parameters[2], parameters[5]),
        color="lightsalmon",
        linestyle='dotted',
        label="Background"
    )
    
    # Extracting signal and its uncertainty the prarmeters. 
    signal_fraction = parameters[0]
    signal_uncertain = np.sqrt(parameter_variance[0, 0])
    if display_signal:
        print(
            "Signal Fraction: % " 
            + str(round(100*signal_fraction, 12)) 
        )
        print(
            "SF Uncertainty: % " 
            + str(round(100*signal_uncertain, 12)) 
        )
    if return_signal_only:
        return signal_fraction, signal_uncertain
    
    # Labeling histogram and corresponding legend. 
    if signal_fraction <= .9994:
        signal_fraction_string = "{:.3f}".format(signal_fraction)[1:]
    else:
        signal_fraction_string = ".999"
    plt.legend(title="Sig. Frac. = "
            + signal_fraction_string
            + "±"
            + "{:.3f}".format(signal_uncertain)[1:])
    plt.savefig("plots\\" + hist_name + ".png")
    return signal_fraction, signal_uncertain


sf=[]
su=[]
iso_rng=[.025*i for i in range(1, 41)]
for iso in iso_rng:
    fraction, uncertainty = fitted_histogram(
        iso_upper_bound=iso,
        display_signal=True,
        return_signal_only=True
    )
    
    plt.clf()
    sf.append(fraction)
    su.append(uncertainty)
    print("")
    print(20*'*')
    print("Isolation upper boundary:", iso)


print("")
print(20*"*")
print(sf)
sf = np.array(sf)
su = np.vstack((
    np.array(su),
    np.array(su)
))

for i,u in enumerate(su[1,:]):
    if sf[i] + u > 1:
        su[1,i] = 1 - sf[i]

print(10*"*")
print("sf:")
print(sf)
print("su:")
print(su)
print("")

plt.errorbar(
    iso_rng, 
    sf,
    yerr=su,
    color='indigo',
    ecolor='plum',
    elinewidth=0.4
)

plt.xlabel("Isolation")
plt.ylabel("Signal Fraction")
plt.title("Signal Fraction vs. Isolation Upper Boundary.")
plt.savefig("plots\\signal_vs_iso.png")