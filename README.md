# entanglement_analysis  
  
Gaussian linker analysis code for finding entanglements in a protein structure  
  
recommended to use a miniconda   

  
---
  
# Packages required to run this script  
  
os, sys, numpy, time, MDAnalysis, scipy, itertools, joblib, configparser, pickle, topoly  
  
some of these are standard python packages and others will need to be installed. you can find how to install them by   
googling the package name and install  
  
  
---
  
# USAGE  

python entanglement_analysis_v1.9.py [1] [2] [3] [4] [5] [6] [7] [8]
  
[1] = path to control file (see explination below)  
[2] = base name for output files (personal tag for the user)  
[3] = start_frame   
[4] = end_frame  
[5] = frame_stride  
[6] = num processors   
[7] = path to referance coordinate file  
[8] = dcd file    
  
### Control File contents  
\[DEFAULT\]  
  
psf = full or relative path to protein structrue file if analysing a trajectory (can be "nan" if only analyzing single PDB for entanglement).  
out_path = full or relative path to an output file (will be made if it doesnt exist) (REQUIRED).  
sec_elems_file_path = full or relative path to secondary structure file if analysing a trajectory (can be "nan" if only analyzing single PDB for entanglement).  
ref_mask = MDAnalysis style atom selection for reference state (should select only CA atoms) (REQUIRED).  
traj_mask = MDAnalysis style atom selection for trajectory (should select only CA atoms) (can be "nan" if only analyzing single PDB for entanglement).  
  
### Secondary Structrue file contents  
column 1 = label (arbitrary and only for user).  
column 2 = start of secondary structure.  
column 3 = end of secondary structure.   
  
AlphaHelix 52 56  
AlphaHelix 60 69  
AlphaHelix 71 77  
AlphaHelix 82 94  
AlphaHelix 109 126  
AlphaHelix 134 144  
AlphaHelix 146 151  
AlphaHelix 160 178  
Strand 8 15  
Strand 25 33  
Strand 39 47  
Strand 104 106  
Strand 131 133  
  
  
---
  
# Example commands:  
  
see examples folder for full examples with example input and output files  
  
1. if you want to just analyze a single PDB   
python entanglement_analysis_v1.7.py single_pdb_anal_inpfiles/single_pdb_anal.cntrl 6u32 0 0 0 4 nan  
  
2. if you want to analyze a trajectory relative to a reference state  
python entanglement_analysis_v1.7.py traj_anal_inpfiles/traj_anal.cntrl dimer 0 10 1 4 traj_anal_inpfiles/cg_dimer_short.dcd  
  
  
---
  
# Outfile contents  
Files 1 and 2 are both output if analyzing a trajector. If only analysing a single PDB only file 2 is output  
1. the time series for fraction of native contacts (Q) and the fraction of native contacts with a change in entanglement (G)  
2. a pickle binary file containing a single dictionary with one entry for the reference state and a single entry for each frame analyzed in the trajectory  
    this file can be examined by launching an interactive python session and using the following commands  
  
    import pickle  
    with open('./test_outfiles/entanglement_analysis_1.4/output/6u32.pkl', 'rb') as fh:  
        data = pickle.load(fh)  
  
    for k,v in data.items():  
        print(k,v)  
  
  
the top level of keys will be integers ranging from -1 to the number of frames analyzed minus 1. For eample if you  
analyzed a trajectory with 10 frames the dictionary would have a total of 11 entries with the following keys  

-1 = reference state results  
0 = first frame results  
1 = second frame results  
...  
9 = tenth frame results  


in each of these entires the value is another dictionary containing one entry for each native contact that was detected to have a change in entanglement  

for the refernce state key = -1 the inner dictionary will be structured like this  
key = (residues invovled in native contact)  
value = [array containing partial linking values for the N and C termini for this native contact,  
         array containing the N and C terminal crossings for this native contact,  
         residues within 8A of the crossing residues]  

so for example if the key value pair in the reference state returned this  
(4, 50) [array([0.        , 0.84160559]), [[], [61]], [[], [24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63]]]  
the native contact formed by the residues 4 and 50 has an entanglement present.  
the partial linking value of the N terminus is 0 indicating no entanglement  
the partial linking value of the C terminus is 0.84160559 indicating a entanglement is present  
Therefor the N terminus should not have a crossing while the C terminus should and indeed does at residue 61  
The residues who have alpha carbons within 8A of the crossing residues are reported last  


for the frames in the traj key >= 0 the inner dictionary will be structured like this  
key = (residues invovled in native contact)  
value = [array containing partial linking values for the N and C termini for this native contact,  
         array containing [change type, refernce state partial linking value, frame partial linking value] for the N and C termini for this native contact,  
         array containing the N and C terminal crossings for this native contact,  
         residues within 8A of the crossing residues]  

change_type = 0 if gain of entanglement and no change in chirality  
change_type = 1 if gain of entanglement and change in chirality  
change_type = 2 if loss of entanglement and no change in chirality  
change_type = 3 if loss of entanglement and change in chirality  

