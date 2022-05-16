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

python entanglement_analysis_v1.4.py [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11]  
  
[1] = path to control file (see explination below)  
[2] = base name for output files (personal tag for the user)  
[3] = start_frame   
[4] = end_frame  
[5] = frame_stride  
[6] = num processors  
[7] = dcd file  
  
### Control File contents  
\[DEFAULT\]  
  
ref_path = full or relative path to PDB or CRD file used for reference state (REQUIRED)  
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
python entanglement_analysis_v1.4.py single_pdb_anal_inpfiles/single_pdb_anal.cntrl 6u32 0 0 0 4 nan  
  
2. if you want to analyze a trajectory relative to a reference state  
python entanglement_analysis_v1.4.py traj_anal_inpfiles/traj_anal.cntrl dimer 0 10 1 4 traj_anal_inpfiles/cg_dimer_short.dcd  
  
  
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


# Automated Entanglement Clustering  

Uses DBSCAN to cluster the resulting .pkl file generated from entanglement_analysis_v1.4.py   
Optimizes the DBSCAN parameters through iteration and then provides summary metrics for the optimal clusters  

The clustering can be applied to a single file or multiple .pkl files in a directory.  

NOTE: Dont cluster across multiple files unless they have the same sequence or are premapped  

# Packages required to run this script  
  
os, sys, numpy, time, scipy, itertools, pickle, sklearn, glob  

some of these are standard python packages and others will need to be installed. you can find how to install them by   
googling the package name and install  

# USAGE  

python automate_change_ent_clustering_v5.1.py [1] [2] [3] [4]   

[1] = path to .pkl input file resulting from entanglement_analysis_v1.4.py  
[2] = path to output file for summary statistics  
[3] = start frame to include in clustering  
[4] = end frame to include in clustering  
  
# Example command  

python automate_change_ent_clustering_v5.1.py inpfiles/nbd1_msm_samples.pkl nbd1_msm_samples 0 29  

# OUTPUT
## File 1(s) 
A summary file for each class of change in entanglement present in the range of frames specified by the use in the .pkl file.  
  
0 = Gain of entanglement  
1 = Gain of entanglement and switch in chirality  
2 = Loss of entanglement  
3 = Loss of entanglement and switch in chirality  
4 = Switch in chirality only  
  

it contains the following columns comma separated  
[1] label: cluster identification label  
[2] clust_size: # NC that have an entanglement present in cluster  
[3] rep_nc: native contacts forming min loop containing all crossings in the cluster.  
[4] rep_crossings: set of unique crossings in the cluster  
[5] rep_surr_res: set of unique residues within 8A of crossing residues in the cluster  
  
## File 2  
a pickle binary file containing a list of the raw change in entanglement data annotated with the clusterid   
    this file can be examined by launching an interactive python session and using the following commands  
    
    import pickle    
    with open('./outfiles/nbd1_msm_samples_total_data.npy', 'rb') as fh:  
        data = pickle.load(fh)  
  
    for k,v in data.items():  
        print(k,v)  
  
Example entry:  
  [0, 1, 24, array([184, 216]), array([ 0.168, -0.782 ]), [0.09958, -0.782125], [[233]], [[69, 70, 184, 185]]]  

[1] = change type as defined above  
[2] = cluster ID  
[3] = frame  
[4] = native contact  
[5] = partial linking values for N and C termini respectfully  
[6] = partial g before and after change   
[7] = crossings  
[8] = residues within 8 A of crossings  

  
# Explination of clustering  

The set of native contacts that have a change in entanglement present in the frames specified will be clustered with the Desnity Based Spatial Clustering of Applications with Noise.  

## Basics of how the algorithm works:  
- obviously a density based algorithm that views the space as populated by areas of high and low densities of points (in this case NC pairs)  
- we first standardize our data by removing the mean and scaling to unit variance.  
- it finds core samples that have a minimum  number of samples surrounding it.  
- then it creates clusters of sets of core samples that share or are them selves common neighbors  
- any points that are greater than eps away from any core sample are considered noise and given a cluster label of -1  
- all other clusters are labeled starting from 0  
  
## Basics of automating DBSCAN  
- there are two metrics the user supplies the algorithm  
    1. eps: the distance threshold between two points to determine if they are in the same neighborhood.  
    2. min_samples: the number of samples in the neighborhood of a point for it to be considered a core_sample.  
- to find the optimal pair of values we iterate of each pair of (eps, min_samples) set of parameters and calculate a clustering quality score  
    - Mean silhouette score = < ( b - a ) / max( a, b ) >s  
        - where a is the mean distance between a sample s and all other points in the same cluster  
        - where b is the mean distance between a sample s and all other points in the next nearest cluster  
- The mean silhouette score reaches a maximimum when the clusters are well defined in space (i.e. little over or under clustering)   
    - therefore the optimal pair of (eps, min_samples) parameters will occure when the mean silhouette score is at a maximum.   
    - for more info about the [mean silhouette score](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)  

## Why is it appropriate to choose DBSCAN for the clustering of native contacts that have entanglement present?  
1. DBSCAN finds clusters of any shape   
2. it can find outliers that may represent very weak entanglements only containing a single native contact forming the loop  
3. it forms clusters in a more intuitive way than basic density clustering  

