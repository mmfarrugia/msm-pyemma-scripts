#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mdshare
import pyemma
from pyemma.util.contexts import settings
import mdtraj as md
import glob
import deeptime
import pickle


# initial resolution 50ps, resolution of 00 pentapeptide tutorial was 0.1 ns so decided on stride of 2 ^ to bring to 0.1 ns resolution


uni_dir = '/scratch365/mfarrugi/HMGR/universal-files'
ts2_dir = '/scratch365/mfarrugi/HMGR/500ns/ts2'
prmtop = md.load_prmtop(uni_dir+'/ts2-strip.prmtop')
ts2_files = glob.glob(ts2_dir+'/align/*-align.dcd')

files = ts2_files
md_timestep = 0.050 # iniital md resolution of 50 ps, or 0.05 ns
md_stride = 4 # load md to featurizers such that feature resolution is 0.2 ns or 200 ps

print(prmtop)
print(files)

feats = pyemma.coordinates.featurizer(prmtop)
heavy = feats.select_Heavy()
ligands_heavy = heavy[heavy < 170]
# NOTE: mdtraj resid is 0-based
ca_hinge_flap_selection = 'name CA and resid 374 to 427'
ca_hinge_flap = feats.select(ca_hinge_flap_selection)

# NOTE: mdtraj resid is 0-based
ligand_contacts_selection = 'name CA and (resid 0 or resid 1 or resid 10 or resid 82 resid 84 or resid 94 or resid 260 or resid 266 or resid 270 or resid 282 or resid 363 or resid 365 or resid 366 or resid 368 or resid 369 or resid 380 or resid 384 or resid 406 or resid 552 or resid 645 or resid 681 or resid 687 or resid 688 or resid 708 or resid 715 or resid 782)'
ca_ligand_contacts = feats.select(ligand_contacts_selection)
feats.add_distances(indices=ligands_heavy, periodic=True, indices2=ca_hinge_flap)
feats.add_distances(indices=ligands_heavy, periodic=True, indices2=ca_ligand_contacts)
feats.add_distances(indices=ca_hinge_flap, periodic=True)
feats.add_distances(indices=ca_hinge_flap, periodic=True, indices2=ca_ligand_contacts)
# second option based on Jinping conclusion + my adjustments
feats2 = pyemma.coordinates.featurizer(prmtop)
feats2.add_distances(indices=ca_hinge_flap, periodic=True)
ligands_and_contacts = [*ligands_heavy, *ca_ligand_contacts]
feats2.add_distances(indices=ligands_and_contacts, periodic=True, indices2=ca_hinge_flap)

labels=['Jinping Pair Distances']#['Ligand Heavy\nCA Hinge Flap\nDistances', 'Ligand Heavy\nCA Ligand Contacts\nDistances', 'CA Hinge Flap\nCA Hinge Flap\nDistances', 'CA Hinge Flap\nCA Ligand Contacts\nDistances']

with open("jl1_ftr_labels.txt", "wb") as wfile:
    pickle.dump(feats.describe(), wfile)
with open("jl2_ftr_labels.txt", "wb") as wfile:
    pickle.dump(feats2.describe(), wfile)

# load in trajectories
#print( files)
distances = pyemma.coordinates.load(files, top=prmtop, features=feats, stride = md_stride)
for i, traj in enumerate(distances):
    np.save('ts2_jl1_ftrs_'+str(i)+'.npy', traj)

print("distances")
print('type of data:', type(distances))
print('lengths:', len(distances))
print('shape of elements:', distances[0].shape)
print('element example: ', distances[0])
print('n_atoms:', feats.topology.n_atoms)
distances2 = pyemma.coordinates.load(files, top=prmtop, features=feats2, stride = md_stride)
for i, traj in enumerate(distances2):
    np.save('ts2_jl2_ftrs_'+str(i)+'.npy', traj)
#np.save('ts2_jl2_ftrs.npy', distances2)
print("distances2")
print('type of data:', type(distances2))
print('lengths:', len(distances2))
print('shape of elements:', distances2[0].shape)
print('element example: ', distances2[0])
print('n_atoms:', feats2.topology.n_atoms)
labels+= ['Jinping Dist 2']

# In[ ]:
torsions_feat = pyemma.coordinates.featurizer(prmtop)
torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
phi_psi = pyemma.coordinates.load(files, features=torsions_feat, stride=md_stride)

for i, traj in enumerate(phi_psi):
    np.save('ts2_phi_psi_ftrs_'+str(i)+'.npy', traj)
#np.save('ts2_phi_psi_ftrs.npy', phi_psi)

with open("phi_psi_ftr_labels.txt", "wb") as wfile:
    pickle.dump(torsions_feat.describe(), wfile)

torsions_feat.add_sidechain_torsions(cossin=True, periodic=False)
torsions_data = pyemma.coordinates.load(files, features=torsions_feat, stride=md_stride)
for i, traj in enumerate(torsions_data):
    np.save('ts2_torsn_ftrs_'+str(i)+'.npy', traj)
#np.save('ts2_torsn_ftrs.npy', torsions_data)
print("torsions")
print('type of data:', type(torsions_data))
print('lengths:', len(torsions_data))
print('shape of elements:', torsions_data[0].shape)
print('element example: ', torsions_data[0])
print('n_atoms:', torsions_feat.topology.n_atoms)
labels += ['torsions'] # backbone and sidechain

positions_feat = pyemma.coordinates.featurizer(prmtop)
positions_feat.add_selection(positions_feat.select_Backbone())
positions_data = pyemma.coordinates.load(files, features=positions_feat, stride=md_stride)
for i, traj in enumerate([positions_data]):
    np.save('ts2_backbone_xyz_ftrs_'+str(i)+'.npy', traj)
#np.save('ts2_backbone_xyz_ftrs.npy', positions_data)
print("backbone xyz")
print('type of data:', type(positions_data))
print('lengths:', len(positions_data))
print('shape of elements:', positions_data[0].shape)
print('element example: ', positions_data[0])
print('n_atoms:', positions_feat.topology.n_atoms)
labels += ['backbone xyz']

with open("torsion_ftr_labels.txt", "wb") as wfile:
    pickle.dump(torsions_feat.describe(), wfile)
with open("backbone_xyz_ftr_labels.txt", "wb") as wfile:
    pickle.dump(positions_feat.describe(), wfile)
