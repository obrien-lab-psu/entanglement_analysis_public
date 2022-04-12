#!/usr/bin/env python3

import numpy as np
import sys,os
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import glob

np.set_printoptions(suppress=False, precision=3, linewidth=10000, threshold=10000)

if len(sys.argv) != 3:
    print(f'[1] = path to .pkl file')
    print(f'[2] = path to summary file')
    quit()

file_path = sys.argv[1]

def clustering(X, eps, min_samples):

    ent_label_array = []
    X = StandardScaler().fit_transform(X)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    components = clustering.components_
    core_sample_indices = clustering.core_sample_indices_

    if len(np.unique(labels)) >= 2 and len(np.unique(labels)) < len(X):
        Qscore = silhouette_score(X, labels)

    else:
        Qscore = -100

    return labels, components, core_sample_indices, Qscore

def load_data(file_path):
    #load data

    files = glob.glob(file_path)

    samples = []
    total_orig_data = {}

    for f_idx,f in enumerate(files):

        with open(f,'rb') as fh:
            orig_data = pickle.load(fh)[-1]
            total_orig_data[f_idx] = orig_data

        num_samples = len(orig_data)

        for i,(k,v) in enumerate(orig_data.items()):
            #print('\n',i,k,v)
            crossings = []
            if abs(v[0][0]) >= 0.7:
                crossings.append(v[1][0])
            if abs(v[0][1]) >= 0.7:
                crossings.append(v[1][1])

            if len(crossings) > 0:
                crossings = np.hstack(crossings)
                num_crossings = len(crossings)
                max_crossing = max(crossings)
                min_crossing = min(crossings)
                median_crossing = np.median(crossings)

                #print(k, crossings, num_crossings, max_crossing, min_crossing, median_crossing)

                samples.append([k[0], k[1], f_idx])


    samples = np.vstack(samples)
    return samples, total_orig_data


#find silhouet trace
quality_est_data = []
#load data
print(f'\n--------------------------------------------------------------')
print(f'LOADING data: {file_path}')
data, orig_data = load_data(file_path)
#print(data)
print(f'\n--------------------------------------------------------------')
print(f'OPTIMIZING DBSCAN eps and min_samples params')
for eps in np.arange(0.1,10,0.1):

    for min_samples in np.arange(1,20,1):
        #print(f'\n--------------------------------------------------------------')

        labels, components, core_sample_indices, Qscore = clustering(data[:,0:2], eps, min_samples)
        #print(f'eps: {eps} | min_samples: {min_samples} | Qscore: {Qscore}')

        quality_est_data.append([eps, min_samples, Qscore])

        if Qscore == -100:
            break
            break

#quality_est_data = np.flipud(np.vstack(quality_est_data))
quality_est_data = np.vstack(quality_est_data)[::-1]
#print(quality_est_data)

sil_opt_params = quality_est_data[np.argmax(quality_est_data[:,2])]
print(f'optimal silhouette score: {sil_opt_params[2]}')
print(f'optimal silhouette score eps: {sil_opt_params[0]}')
print(f'optimal silhouette score min_samples: {sil_opt_params[1]}')

labels, components, core_sample_indices, Qscore = clustering(data[:,0:2], sil_opt_params[0], sil_opt_params[1])
#print(f'all labels: {labels}')
#print(f'components: {components}')
#print(f'core_sample_indices: {core_sample_indices}')
print(f'\n--------------------------------------------------------------')
print('SUMMARY of custers')
total_sample_size = data.shape[0]
outdata = []
for label in np.unique(labels):
    #print(f'\n\033[97mlabel: {label}')
    cidx = np.where(labels == label)[0]
    cdata = data[cidx,:]

    label_nc = []
    label_gvals = []
    label_crossings = []
    label_surr_res = []
    for nc in cdata:
        file_idx = nc[2]
        ent_info = orig_data[file_idx][tuple(nc[0:2])]

        nc = np.asarray(nc[0:2])
        crossings = []
        gvals = []
        if abs(ent_info[0][0]) >= 0.7:
            crossings.append(ent_info[1][0])
            gvals.append(ent_info[0][0])
        if abs(ent_info[0][1]) >= 0.7:
            crossings.append(ent_info[1][1])
            gvals.append(ent_info[0][1])
        crossings = np.hstack(crossings).astype(int)
        surr_res = np.hstack(ent_info[2]).astype(int)

        label_nc.append(nc)
        label_gvals.append(gvals)
        label_crossings.append(crossings)
        label_surr_res.append(surr_res)

        #print(f'\n\033[97mLabel: {label}\nfile_idx: {file_idx}\n\033[91mNative Contact: {" ".join(nc)}\n\033[96mGaussLink_vals: {" ".join(gvals)}\n\033[93mCrossings: {" ".join(crossings)}\n\033[92mSurrounding Residues: {" ".join(surr_res)}')
        #print(nc_idx,nc, crossings, surr_res)


    num_nc = len(label_nc)
    frac_nc = num_nc / total_sample_size
    #total_sample_size += num_nc
    label_nc = np.unique(np.hstack(label_nc))
    rep_nc = [min(label_nc), max(label_nc)]
    label_nc = label_nc.astype(str)

    label_gvals = np.unique(np.hstack(label_gvals))
    rep_gval = np.mean(gvals)
    label_gvals = label_gvals.astype(str)

    label_crossings = np.unique(np.hstack(label_crossings))
    label_depths = []
    for crossing in label_crossings:
        if crossing < int(rep_nc[0]):
            depth = 1 - abs(0.5 - (crossing / int(rep_nc[0])))
            label_depths.append(depth)

        elif crossing > int(rep_nc[1]):
            depth = 1- abs(0.5 - (crossing / (293-rep_nc[1])))
            label_depths.append(depth)

    label_crossings = label_crossings.astype(str)
    avg_ent_depth = np.mean(label_depths)
    label_surr_res = np.unique(np.hstack(label_surr_res)).astype(str)
    outdata.append([label, frac_nc, avg_ent_depth, (frac_nc+avg_ent_depth)/2, " ".join(np.asarray(rep_nc).astype(str)), " ".join(label_crossings), " ".join(label_surr_res)])

    print(f'\n\033[97mLabel: {label}\n\033[91mrep_nc: {rep_nc}\nnum_nc: {num_nc}\n\033[96mrep_gval: {rep_gval}\n\033[93mcrossings: {" ".join(label_crossings)}\navg_ent_depth: {avg_ent_depth}\n\033[92msurrounding Residues: {" ".join(label_surr_res)}')
print(total_sample_size)

header = 'label, clust_size, avg_ent_depth, score, rep_nc, rep_crossings, rep_surr_res'
outdata = np.asarray(outdata)
np.savetxt(sys.argv[2], outdata, header=header, fmt='%s',delimiter=', ')
print(f'\n\033[97mSAVED: {sys.argv[2]}')
print('\033[97mNORMAL TERMINATION')

