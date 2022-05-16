import numpy as np
import pandas as pd


## Defining function that calculates invariant mass.
def invariant_mass(pT1,phi1,eta1,pT2,phi2,eta2):
    
    # Electron mass in GeV
    mass_e = .510998950 * (10**-3)
    
    # Momentum Vectors.
    p1 = [ pT1*np.cos(phi1), pT1*np.sin(phi1), pT1*np.sinh(eta1) ]
    p2 = [ pT2*np.cos(phi2), pT2*np.sin(phi2), pT2*np.sinh(eta2) ]
    pdot = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
    
    # Energies.
    E1 = np.sqrt( (mass_e**2) + (p1[0]**2) + (p1[1]**2) + (p1[2]**2) )
    E2 = np.sqrt( (mass_e**2) + (p2[0]**2) + (p2[1]**2) + (p2[2]**2) )
    
    # Invariant Mass Calulation.
    return np.sqrt( 2*(mass_e**2 + E1*E2 - pdot) )


## Calcualating invariant mass and writing to csv.
muon_df = pd.read_csv(open("data\\mini_muons.csv",'r'))
imass_df = invariant_mass( pT1 = muon_df.loc[:,"Momenta 1"],
                           phi1 = muon_df.loc[:,"Phi 1"],
                           eta1 = muon_df.loc[:,"Eta 1"],
                           pT2 = muon_df.loc[:,"Momenta 2"],
                           phi2 = muon_df.loc[:,"Phi 2"],
                           eta2 = muon_df.loc[:,"Eta 2"])
imass_df.to_csv("data\\invariant_mass.csv", header = ["Invariant Mass"])
    