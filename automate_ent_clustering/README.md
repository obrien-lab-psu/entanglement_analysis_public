# Automated Entanglement Clustering  

Uses DBSCAN to cluster the resulting .pkl file generated from entanglement_analysis_v1.4.py   
Optimizes the DBSCAN parameters through iteration and then provides summary metrics for the optimal clusters  

The clustering can be applied to a single file or multiple .pkl files in a directory.  

NOTE: Dont cluster across multiple files unless they have the same sequence or are premapped  

# USAGE  

python automate_ent_clustering_v4.3.py [1] [2]  

[1] = path to .pkl input file resulting from entanglement_analysis_v1.4.py  
[2] = path to output file for summary statistics  

# Example command  

python automate_ent_clustering_v4.3.py 6u32_CP4.pkl 6u32_CP4.summary  

# OUTPUT

A summary file for each cluster resulting from the optimized DBSCAN is created at the specified output file  
it contains the following columns comma separated  

[1] label: cluster identification label  
[2] clust_size: # NC that have an entanglement present in cluster / total # NC in while protein with entanglement present  
[3] avg_ent_depth: average of the distribution of depths calculated for each crossing found in the cluster. the depth is defined by the following:  
- 1 - abs(0.5 - (l/Lt))  
    - where l = the minimal number of residues between the native contact closing the loop and the crossing  
    - where Lt = the length of the thread  

[4] score = (clust_size + avg_ent_depth) / 2  
[5] rep_nc: min and max of set of native contacts forming loops in the cluster. This ensures you get the largest loop encompasing all crossings in the cluster.    
[6] rep_crossings: set of unique crossings in the cluster  
[7] rep_surr_res: set of unique residues within 8A of crossing residues in the cluster  


# Explination of clustering
