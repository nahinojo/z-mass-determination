### Converts the .txt data to .csv. 
## Importing libraries.
import pandas as pd

# Instantiating file and datafram objects. 
file_txt = open("data\\mini_muons.txt",'r')
df = pd.DataFrame(columns = ["Momenta 1","Momenta 2","Eta 1","Eta 2","Phi 1","Phi 2","Iso 1","Iso 2"])

## Extracts data in .txt file line-by-line.
investigate_collision = 0
lines_read = 0
for line_index, line in enumerate(file_txt):
    # Scans for instances where the number of electrons is two.
    if "NumMuons: 2" in line:    
        # Initial variables for data extraction. 
        investigate_collision = 2
        momenta = []
        eta = []
        phi = []
        charge = 0
        iso = []
    
    # Will extract collision data for two iterations.
    elif investigate_collision > 0:
        # Noting charges
        if line[7 + line.find("Charge ")] == '1':
            charge += 1
        else:
            charge -= 1

        # On the last iteration the charges should sum to zero. 
            # If not, ignore this case. 
        if investigate_collision == 1 and charge != 0:
            investigate_collision = 0
            continue

        # Extracting Momenta.
        index_l = line.find("Pt") + 3
        index_r = line.find(" Eta")
        momenta.append(float(line[index_l:index_r]))

        # Extracting Eta.
        index_l=5+index_r
        index_r=line.find(" Phi")
        eta.append(float(line[index_l:index_r]))

        # Extracting Phi.
        index_l=5+index_r
        index_r=line.find(" Charge")
        phi.append(float(line[index_l:index_r]))

        # Extracting Iso.
        index_l = line.find("Iso") + 4
        index_r = len(line) - 1
        iso.append(float(line[index_l:index_r]))
        
        # Investigates collision twice. 
        investigate_collision-=1

        # Appends extrated data to dataframe. 
        if investigate_collision==0:
            df.loc[len(df.index)]=[momenta[0],momenta[1],eta[0],eta[1],phi[0],phi[1],iso[0],iso[1]]

## Writes the panda dataframe as a .csv.
df.to_csv("data\\mini_muons.csv")