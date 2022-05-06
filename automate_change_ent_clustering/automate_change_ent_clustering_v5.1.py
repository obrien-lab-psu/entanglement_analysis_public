#!/usr/bin/env python3

import numpy as np
import sys,os
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import glob

### this version supports multiple frames but requires the user to specify

np.set_printoptions(suppress=False, precision=3, linewidth=10000, threshold=10000)

if len(sys.argv) != 5:
    print(f'[1] = path to .pkl file')
    print(f'[2] = path to summary file')
    print(f'[3] = start frame to analyze')
    print(f'[4] = end frame to analyze')
    quit()

file_path = sys.argv[1]
start = int(sys.argv[3])
end = int(sys.argv[4])

def clustering(X, eps, min_samples):

    ent_label_array = []
    #X = StandardScaler().fit_transform(X)
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

    samples = {0:[], 1:[], 2:[], 3:[], 4:[]}
    total_orig_data = {}

    for f_idx,f in enumerate(files):
        print(f_idx, f)
        with open(f,'rb') as fh:

            orig_data = pickle.load(fh)
            total_orig_data[f_idx] = orig_data

        change_flags = []
        for frame in range(start, end+1):
            #print(frame)
            for i,(k,v) in enumerate(orig_data[frame].items()):
                #print('\n',i,k,v)
                crossings = []
                changes = []

                if len(v[1][0]) > 0:
                    change_flags += [v[1][0][0]]
                    changes += [v[1][0][0]]
                    crossings += v[2]

                if len(v[1][1]) > 0:
                    change_flags += [v[1][1][0]]
                    changes += [v[1][1][0]]
                    crossings += v[2]

                if crossings:
                    crossings = np.hstack(crossings)
                    crossings = np.hstack(crossings)
                    num_crossings = len(crossings)
                    max_crossing = max(crossings)
                    min_crossing = min(crossings)
                    median_crossing = np.median(crossings)

                    #print(k, crossings, num_crossings, max_crossing, min_crossing, median_crossing)
                    for change in changes:
                        samples[change] += [[k[0], k[1], min_crossing, max_crossing, frame, f_idx]]

    if len(samples) == 0:
        print(f'No native contacts had an entanglement present, Exitting...')
        quit()
    return samples, total_orig_data


#load data
print(f'\n--------------------------------------------------------------')
print(f'LOADING data: {file_path}')
sdata, orig_data = load_data(file_path)

total_outdata = []
for change_type in [0,1,2,3,4]:
    quality_est_data = []
    print(f'\nCHANGE_TYPE: {change_type}')

    data = np.asarray(sdata[change_type])
    if data.size == 0:
        continue

    print(f'\n--------------------------------------------------------------')
    print(f'OPTIMIZING DBSCAN eps and min_samples params')
    for eps in np.arange(1,30,1):

        #for min_samples in np.arange(1,20,1):
        for min_samples in np.arange(3,20,1):
            #print(f'\n--------------------------------------------------------------')

            labels, components, core_sample_indices, Qscore = clustering(data[:,2:4], eps, min_samples)
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

    labels, components, core_sample_indices, Qscore = clustering(data[:,2:4], sil_opt_params[0], sil_opt_params[1])
    #print(f'all labels: {labels}')
    #print(f'components: {components}')
    #print(f'core_sample_indices: {core_sample_indices}')
    print(f'\n--------------------------------------------------------------')
    print('SUMMARY of custers')
    total_sample_size = data.shape[0]
    outdata = []
    clustered_outdata = {}
    frame_outdata = []
    for label in sorted(np.unique(labels)):
        print(f'\n\033[97mlabel: {label}')
        cidx = np.where(labels == label)[0]
        cdata = data[cidx,:]


        label_nc = []
        label_diff = []
        label_gvals = []
        label_crossings = []
        label_surr_res = []
        label_frames = []
        for nc in cdata:
            file_idx = nc[-1]
            frame = nc[-2]
            label_frames.append(frame)
            ent_info = orig_data[file_idx][frame][tuple(nc[0:2])]

            nc = np.asarray(nc[0:2])

            crossings = []
            gvals = []
            chnginfo = []

            if len(ent_info[1][0]) > 0:
                crossings.append(ent_info[2])
                gvals.append(ent_info[0][0])
                chnginfo = ent_info[1][0][1:3]

            if len(ent_info[1][1]) > 0:
                crossings.append(ent_info[2])
                gvals.append(ent_info[0][1])
                chnginfo = ent_info[1][1][1:3]

            crossings = np.hstack(crossings).astype(int)
            surr_res = np.hstack(ent_info[3]).astype(int)

            print(change_type, label, frame, nc, ent_info[0], chnginfo, ent_info[2], ent_info[3])
            total_outdata.append([change_type, label, frame, nc, ent_info[0], chnginfo, ent_info[2], ent_info[3]])

            label_nc.append(nc)
            label_diff.append(np.diff(nc))
            label_gvals.append(gvals)
            label_crossings.append(crossings)
            label_surr_res.append(surr_res)

        label_nc = np.vstack(label_nc)
        label_diff = np.hstack(label_diff)

        #find nc with minimal loop that encompases all crossings
        sorted_idx = np.argsort(label_diff)
        u_crossings = np.unique(np.hstack(label_crossings))
        max_jscore = 0
        for idx, nc in enumerate(label_nc[sorted_idx]):
            crossing = label_crossings[sorted_idx[idx]]

            set1 = set(u_crossings)
            set2 = set(np.arange(min(crossing[0]-3), max(crossing[0])+3))

            jscore = len(set1.intersection(set2))/len(set1.union(set2))

            if jscore > max_jscore:
                max_jscore = jscore
                rep_nc = nc

        num_nc = len(label_nc)
        frac_nc = num_nc / total_sample_size
        label_nc = np.unique(label_nc)
        label_nc = label_nc.astype(str)

        label_gvals = np.unique(np.hstack(label_gvals))
        rep_gval = np.mean(gvals)
        label_gvals = label_gvals.astype(str)

        label_crossings = np.unique(np.hstack(label_crossings))
        label_crossings = label_crossings.astype(str)
        label_surr_res = np.unique(np.hstack(label_surr_res)).astype(str)
        outdata.append([label, num_nc, " ".join(np.asarray(rep_nc).astype(str)), " ".join(label_crossings), " ".join(label_surr_res)])

    #processes data for output
    header = 'label, clust_size, rep_nc, rep_crossings, rep_surr_res'
    outdata = np.asarray(outdata)

    #save summary outdata
    outdata = np.asarray(outdata)

    outdata = outdata[outdata[:,1].astype(int).argsort()]
    for d in outdata:

        print(f'\n\033[97mLabel: {d[0]}\nclust_size: {d[1]}\n\033[91mrep_nc: {d[2]}\n\033[93mcrossings: {d[3]}\n\033[92msurrounding Residues: {d[4]}')

    np.savetxt(f'{sys.argv[2]}_{change_type}.summary', outdata, header=header, fmt='%s',delimiter=', ')
    print(f'\n\033[97mSAVED: {sys.argv[2]}_{change_type}.summary')

for o in total_outdata:
    print(o)

total_outdata = np.asarray(total_outdata, dtype='O')
np.save(f'{sys.argv[2]}_total_data',total_outdata)
print(f'SAVED: {sys.argv[2]}_total_data.npy')
print('\033[97mNORMAL TERMINATION')

