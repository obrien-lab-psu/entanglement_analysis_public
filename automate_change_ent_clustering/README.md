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

For a good explination of [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)  
