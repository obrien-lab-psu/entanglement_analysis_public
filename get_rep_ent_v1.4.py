#!/usr/bin/env python3

import numpy as np
import sys,os
import matplotlib
import glob
import pickle
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

np.set_printoptions(precision=3)

if len(sys.argv) != 6:
    print('[1] path to pkl file')
    print('[2] start frame')
    print('[3] end frame')
    print('[4] rep ent outfile name must end with .csv')
    print('[5] raw cluster binary pkl file output file. must end with .pkl')
    sys.exit()


def load_data(file_path):
    #load data
    for frame in range(start, end+1):
        #print(frame)

        for i,(k,v) in enumerate(orig_data[frame].copy().items()):
            crossings = []
            surr = []

            if frame != -1:
                #print(i,k,v)

                #Nterminal changbe
                if len(v[1][0]) > 0:

                    change_type = v[1][0][0]
                    ref_g = v[1][0][1]
                    traj_g = v[1][0][2]
                    #print(change_type, ref_g, traj_g)

                    if change_type in [0,1]: #gain
                        if np.abs(traj_g) < 0.6:
                            orig_data[frame][k][1][0] = np.asarray([])
                        else:
                            orig_data[frame][k][1][0] = np.asarray(orig_data[frame][k][1][0])
                            crossings += v[2]
                            surr += v[-1]

                    elif change_type in [2,3]: #loss
                        if np.abs(ref_g) < 0.6:
                            orig_data[frame][k][1][0] = np.asarray([])
                        else:
                            orig_data[frame][k][1][0] = np.asarray(orig_data[frame][k][1][0])
                            crossings += v[2]
                            surr += v[-1]

                    elif change_type in [4]: #switch chirl
                        if np.abs(traj_g) < 0.6 or np.abs(ref_g) < 0.6:
                            orig_data[frame][k][1][0] = np.asarray([])
                        else:
                            orig_data[frame][k][1][0] = np.asarray(orig_data[frame][k][1][0])
                            crossings += v[2]
                            surr += v[-1]
                else:
                    orig_data[frame][k][1][0] = np.asarray([])


                if len(v[1][1]) > 0:

                    change_type = v[1][1][0]
                    ref_g = v[1][1][1]
                    traj_g = v[1][1][2]
                    #print(change_type, ref_g, traj_g)

                    if change_type in [0,1]: #gain
                        if np.abs(traj_g) < 0.6:
                            orig_data[frame][k][1][1] = np.asarray([])
                        else:
                            orig_data[frame][k][1][1] = np.asarray(orig_data[frame][k][1][1])
                            v[1][1] = np.asarray(v[1][1])
                            crossings += v[2]
                            surr += v[-1]

                    if change_type in [2,3]: #loss
                        if np.abs(ref_g) < 0.6:
                            orig_data[frame][k][1][1] = np.asarray([])
                        else:
                            orig_data[frame][k][1][1] = np.asarray(orig_data[frame][k][1][1])
                            crossings += v[2]
                            surr += v[-1]

                    if change_type in [4]: #switch chirl
                        if np.abs(traj_g) < 0.6 or np.abs(ref_g) < 0.6:
                            orig_data[frame][k][1][1] = np.asarray([])
                        else:
                            orig_data[frame][k][1][1] = np.asarray(orig_data[frame][k][1][1])
                            crossings += v[2]
                            surr += v[-1]

                else:
                    orig_data[frame][k][1][1] = np.asarray([])

            #check if this native contact should be ignored
            if (orig_data[frame][k][1][0].size == 0 and orig_data[frame][k][1][1].size == 0) or (len(crossings) == 0):
                del orig_data[frame][k]

    return orig_data


def cluster(data):

    cluster_dict = {}
    cluster_rep_dict = {}

    frames = np.arange(start, end+1)
    print(f'frames to cluster: {frames}')

    for frame in frames:
        if frame != -1 and frame in data:

            ent_data = data[frame]
            #print(frame, ent_data)
            #if frame not in cluster_dict:
            #    cluster_dict[frame] = {}

            for nc, ent in ent_data.items():
                #print('\n', frame, nc, ent)
                #84, 123) [array([0.73461379, 0.11097391]), [[0, -0.3360524409177605, 0.7346137865365975], []], [[44]], [[41, 42, 43, 45, 46, 90, 91, 92, 113, 114, 115, 117, 118, 121]]]

                if len(ent[2]) == 0:
                    continue
                else:
                    num_cross = len(ent[2][0])

                gN = np.round(ent[0][0]).astype(int)
                gC = np.round(ent[0][1]).astype(int)

                if len(ent[1][0]) > 0:
                    changeN = ent[1][0][0]
                else:
                    changeN = -1

                if len(ent[1][1]) > 0:
                    changeC = ent[1][1][0]
                else:
                    changeC = -1

                key = (num_cross, gN, gC, changeN, changeC)
                #print(key)
                if key not in cluster_dict:
                    cluster_dict[key] = {}

                if frame not in cluster_dict[key]:
                    cluster_dict[key][frame] = {}

                cluster_dict[key][frame][nc] = ent

    # make representative entanglement for each cluster in each frame
    print('# make representative entanglement for each cluster in each frame')
    rep_dict = {}
    #print(cluster_dict.keys())

    for key,frames in cluster_dict.items():
        print(f'KEY: {key}')
        #print(frames)
        rep_dict[key] = {}


        min_loop = 99999
        for frame,ents in frames.items():
            print(f'FRAME: {frame}')
            for nc, ent in ents.items():
                print(nc, ent)

                loop_l = np.diff(nc)[0]
                #print(loop_l)

                if loop_l < min_loop:
                    min_loop = loop_l
                    rep_nc = nc
                    rep_ent = ent
                    rep_frame = frame

        rep_dict[key][rep_nc] = [rep_frame, rep_ent]

    return cluster_dict, rep_dict

#load user arguments
global tart, end, aID
start = int(sys.argv[2])
end = int(sys.argv[3])
outfile = sys.argv[4]
pkl_outfile = sys.argv[5]
start_time = time.time()

files = glob.glob(sys.argv[1])[0]


with open(files,'rb') as fh:
    orig_data = pickle.load(fh)

#check if end supplied is greater than the number of frames
if end > len(orig_data):
    print(f'frames in {files}')
    print(orig_data.keys())
    num_frames = len([x for x in orig_data.keys() if x != -1])
    end_frame = max(orig_data.keys())
    print(f'supplied end frame {end} > amount of frames in .pkl file {num_frames}\nsetting end={end_frame}')
    end = end_frame

#load in orgiinal data
orig_data = load_data(orig_data)

#cluster changes in ent
cluster_data, rep_data = cluster(orig_data)
with open(f'{pkl_outfile}', 'wb') as fh:
    pickle.dump(cluster_data, fh)
print(f'SAVED: {pkl_outfile}')


print('--------------------------------------------------------------------------------------------------------------')
print('\nRepresentative changes in ent from clustering')
outdict = {'rep_ent_idx': [], 'frame': [], 'NC': [], 'Nchange': [], 'Nchange_ref_g': [], 'Nchange_frame_g': [], 'Cchange': [], 'Cchange_ref_g': [], 'Cchange_frame_g': [], 'crossing': [], 'surrounding': []}
for rep_i, (key, ent_info) in enumerate(rep_data.items()):
    #(#cross, Nchir, Cchir, Nchange, Cchange)
    print(f'\nRep Ent Number: {rep_i}')
    print(f'Number of Crossings: {key[0]}')
    print(f'N ent change chirality: {key[1]}')
    print(f'C ent change chirality: {key[2]}')
    print(f'N ent change type: {key[3]}')
    print(f'C ent change type: {key[4]}')
    for k,(frame, ((gNvalue, gCvalue), (gNchange, gCchange), crossings, surr)) in ent_info.items():
        #print(k, frame, gNvalue, gCvalue, gNchange, gCchange , crossings, surr)
        print(f'Frame: {frame}')
        print(f'Native contact: {k}')
        outdict['rep_ent_idx'] += [rep_i]
        outdict['frame'] += [frame]
        outdict['NC'] += [k]

        if gNchange.size > 0:
            print(f'N-term frame partial linking value: {gNvalue}')
            print(f'N-term ref partial linking value: {gNchange[1]}')
            outdict['Nchange'] += [key[3]]
            outdict['Nchange_ref_g'] += [gNchange[1]]
            outdict['Nchange_frame_g'] += [gNvalue]
        else:
            outdict['Nchange'] += [np.nan]
            outdict['Nchange_ref_g'] += [np.nan]
            outdict['Nchange_frame_g'] += [np.nan]

        if gCchange.size > 0:
            print(f'C-term frame partial linking value: {gCvalue}')
            print(f'C-term ref partial linking value: {gCchange[1]}')
            outdict['Cchange'] += [key[4]]
            outdict['Cchange_ref_g'] += [gCchange[1]]
            outdict['Cchange_frame_g'] += [gCvalue]
        else:
            outdict['Cchange'] += [np.nan]
            outdict['Cchange_ref_g'] += [np.nan]
            outdict['Cchange_frame_g'] += [np.nan]

        print(f'Crossings: {crossings}')
        print(f'Residues with 8A of crossings: {surr}')

        outdict['crossing'] += [crossings]
        outdict['surrounding'] += [surr]

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
df = pd.DataFrame(data=outdict)
print(df)
df.to_csv(outfile, float_format='%.3f')


print('NORAML TERMINATION @ {time.time() - start_time}')
