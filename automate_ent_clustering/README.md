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
