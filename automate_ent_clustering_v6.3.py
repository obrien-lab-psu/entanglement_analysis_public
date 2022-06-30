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

    samples = {-1:[], 0:[], 1:[], 2:[], 3:[], 4:[]}
    total_orig_data = {}

    for f_idx,f in enumerate(files):
        #print(f_idx, f)
        with open(f,'rb') as fh:

            orig_data = pickle.load(fh)
            total_orig_data[f_idx] = orig_data
        for frame in range(start, end+1):
            #print(frame)
            for i,(k,v) in enumerate(orig_data[frame].items()):
                #print('\n',i,k,v)
                crossings = []
                changes = []


                if frame != -1:
                    if len(v[1][0]) > 0:
                        changes += [v[1][0][0]]
                        crossings += v[2]

                    if len(v[1][1]) > 0:
                        changes += [v[1][1][0]]
                        crossings += v[2]

                elif frame == -1:
                    if len(v[1][0]) > 0:
                        changes += [-1]
                        crossings += v[1][0]

                    if len(v[1][1]) > 0:
                        changes += [-1]
                        crossings += v[1][1]

                if crossings:
                    crossings = np.hstack(crossings)
                    crossings = np.hstack(crossings)
                    num_crossings = len(crossings)
                    max_crossing = max(crossings)
                    min_crossing = min(crossings)
                    median_crossing = np.median(crossings)

                    #print(k, crossings, num_crossings, max_crossing, min_crossing, median_crossing)
                    for change in changes:
                        samples[change] += [[k[0], k[1], median_crossing, frame, f_idx]]

    if len(samples) == 0:
        print(f'No native contacts had an entanglement present, Exitting...')
        quit()
    return samples, total_orig_data


#load data
print(f'\n--------------------------------------------------------------')
print(f'LOADING data: {file_path}')
sdata, orig_data = load_data(file_path)
print(sdata)
total_outdata = []
for change_type in [-1,0,1,2,3,4]:
    quality_est_data = []
    print(f'\nCHANGE_TYPE: {change_type}')

    data = np.asarray(sdata[change_type])
    if data.size == 0:
        continue

    print(f'\n--------------------------------------------------------------')
    print(f'OPTIMIZING DBSCAN eps and min_samples params')
    #eps_range = np.arange(3,100,1)
    eps_range = np.asarray([55])
    print(f'eps_range: {eps_range} {len(eps_range)}')
    #min_samples_range = np.arange(5,10,1Vy)
    min_samples_range = np.asarray([5])
    print(f'min_samples_range: {min_samples_range} {len(min_samples_range)}')

    for eps in eps_range:

        for min_samples in min_samples_range:

            labels, components, core_sample_indices, Qscore = clustering(data[:,0:3], eps, min_samples)

            quality_est_data.append([eps, min_samples, Qscore])


    #quality_est_data = np.flipud(np.vstack(quality_est_data))
    quality_est_data = np.vstack(quality_est_data)
    quality_est_data = quality_est_data[:,2].reshape((len(eps_range), len(min_samples_range)))

    max_pos = np.unravel_index(quality_est_data.argmax(), quality_est_data.shape)
    max_score = quality_est_data[max_pos]
    opt_eps = eps_range[max_pos[0]]
    opt_min_samples = min_samples_range[max_pos[1]]

    print(f'optimal silhouette score: {max_score}')
    print(f'optimal silhouette score eps: {opt_eps}')
    print(f'optimal silhouette score min_samples: {opt_min_samples}')

    labels, components, core_sample_indices, Qscore = clustering(data[:,0:3], opt_eps, opt_min_samples)

    if opt_eps == min(eps_range):
        print('Potential Single cluster detected')
        print(labels, labels.shape)
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

            if frame != -1:
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

                #print(change_type, label, frame, nc, ent_info[0], chnginfo, ent_info[2], ent_info[3])
                total_outdata.append([change_type, label, frame, nc, ent_info[0], chnginfo, ent_info[2], ent_info[3]])

            elif frame == -1:
                if len(ent_info[1][0]) > 0:
                    chnginfo = -1
                    gvals.append(ent_info[0][0])
                    crossings += ent_info[1][0]

                if len(ent_info[1][1]) > 0:
                    changes = -1
                    gvals.append(ent_info[0][1])
                    crossings += ent_info[1][1]


                crossings = np.hstack(crossings).astype(int)
                surr_res = np.hstack(ent_info[2]).astype(int)

                #print(change_type, label, frame, nc, ent_info[0], chnginfo, ent_info[1], ent_info[2])
                total_outdata.append([change_type, label, frame, nc, ent_info[0], chnginfo, ent_info[1], ent_info[2]])

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
            crossing = np.unique(np.hstack(label_crossings[sorted_idx[idx]]))
            print(idx, nc, crossing)
            set1 = set(u_crossings)
            set2 = set(np.arange(min(crossing-3), max(crossing)+3))

            jscore = len(set1.intersection(set2))/len(set1.union(set2))

            if jscore > max_jscore:
                max_jscore = jscore
                rep_nc = nc
                rep_cross = crossing
                rep_frame =  label_frames[sorted_idx[idx]]
                rep_surr = label_surr_res[sorted_idx[idx]]

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
        #outdata.append([label, num_nc, " ".join(np.asarray(rep_nc).astype(str)), " ".join(label_crossings), " ".join(label_surr_res)])
        outdata.append([label, num_nc, " ".join(np.asarray(rep_nc).astype(str)), " ".join(rep_cross.astype(str)), " ".join(rep_surr.astype(str)), str(rep_frame)])

    #processes data for output
    header = 'label, clust_size, rep_nc, rep_crossings, rep_surr_res, rep_frame'
    outdata = np.asarray(outdata)

    #save summary outdata
    outdata = np.asarray(outdata)

    outdata = outdata[outdata[:,1].astype(int).argsort()]
    for d in outdata:

        print(f'\n\033[97mLabel: {d[0]}\nclust_size: {d[1]}\n\033[91mrep_nc: {d[2]}\n\033[93mcrossings: {d[3]}\n\033[92msurrounding Residues: {d[4]}\n\033[97mrep_frame: {d[5]}')

    np.savetxt(f'{sys.argv[2]}_{change_type}.summary', outdata, header=header, fmt='%s',delimiter=', ')
    print(f'\n\033[97mSAVED: {sys.argv[2]}_{change_type}.summary')


total_outdata = np.asarray(total_outdata, dtype='O')
np.save(f'{sys.argv[2]}_total_data',total_outdata)
print(f'SAVED: {sys.argv[2]}_total_data.npy')
print('\033[97mNORMAL TERMINATION')

