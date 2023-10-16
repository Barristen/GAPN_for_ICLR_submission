import numpy as np
from biopandas.pdb import PandasPdb
import torch
from scipy.spatial.transform import Rotation
def residue_type_pipr(residue):

    dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',

           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    rare_residues = {'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    if residue in rare_residues.keys():
        print('Some rare residue: ', residue)

    indicator = {'Y': [0.27962074,-0.051454283,0.114876375,0.3550331,1.0615551,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'R': [-0.15621762,-0.19172126,-0.209409,0.026799612,1.0879921,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'F': [0.2315121,-0.01626652,0.25592703,0.2703909,1.0793934,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'G': [-0.07281224,0.01804472,0.22983849,-0.045492448,1.1139168,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'I': [0.15077977,-0.1881559,0.33855876,0.39121667,1.0793937,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'V': [-0.09511698,-0.11654304,0.1440215,-0.0022315443,1.1064949,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'A': [-0.17691335,-0.19057421,0.045527875,-0.175985,1.1090639,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'W': [0.25281385,0.12420933,0.0132171605,0.09199735,1.0842415,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'E': [-0.06940994,-0.34011552,-0.17767446,0.251,1.0661993,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0], 
    'H': [0.019046513,-0.023256639,-0.06749539,0.16737276,1.0796973,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'C': [-0.31572455,0.38517416,0.17325026,0.3164464,1.1512344,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0], 
    'N': [0.41597384,-0.22671205,0.31179032,0.45883527,1.0529875,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
    'M': [0.06302169,-0.10206237,0.18976009,0.115588315,1.0927621,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'D': [0.00600859,-0.1902303,-0.049640052,0.15067418,1.0812483,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0], 
    'T': [0.054446213,-0.16771607,0.22424258,-0.01337227,1.0967118,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'S': [0.17177454,-0.16769698,0.27776834,0.10357749,1.0800852,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
    'K': [0.22048187,-0.34703028,0.20346786,0.65077996,1.0620389,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0], 
    'L': [0.0075188675,-0.17002057,0.08902198,0.066686414,1.0804346,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], 
    'Q': [0.25189143,-0.40238172,-0.046555642,0.22140719,1.0362468,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0], 
    'P': [0.017954966,-0.09864355,0.028460773,-0.12924117,1.0974121,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]}
    res_name = residue
    if res_name not in dit.keys():
        # print('UNK')
        return indicator['H']
    else:
        res_name = dit[res_name]
        return indicator[res_name]
    
def get_residues(pdb_filename):
    residue_list_all = []
    drop_list = []
    res_name_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR',
           'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ',
           'TYS', 'CYX', 'GLZ']
    df = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    drop_list = list(set([item for item in df['residue_name'] if item not in set(res_name_list)]))
    for ele in drop_list:
        df = df.drop(index=df[df['residue_name']==ele].index)
    chain_num = df['chain_id'].unique().shape[0]
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    group = df.groupby('chain')
    for key, value in group:
        residue_list = list(value.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
        residue_list_all.append(residue_list)
    return residue_list_all, chain_num

def filter_residues(residues):
        residues_filtered = []
        for residue in residues:
            df = residue[1]
            Natom = df[df['atom_name'] == 'N']
            alphaCatom = df[df['atom_name'] == 'CA']
            Catom = df[df['atom_name'] == 'C']

            if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
                residues_filtered.append(residue)
        return residues_filtered
def UniformRotation_Translation(translation_interval):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_interval)
    t = t * length
    return rotation_matrix.astype(np.float32), t.astype(np.float32)
def get_alphaC_loc_array(bound_predic_clean_list):
    bound_alphaC_loc_clean_list = []
    for residue in bound_predic_clean_list:
        df = residue[1]
        alphaCatom = df[df['atom_name'] == 'CA']
        alphaC_loc = alphaCatom[['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32)
        assert alphaC_loc.shape == (3,), \
            f"alphac loc shape problem, shape: {alphaC_loc.shape} residue {df} resid {df['residue']}"
        bound_alphaC_loc_clean_list.append(alphaC_loc)
    if len(bound_alphaC_loc_clean_list) <= 1:
        bound_alphaC_loc_clean_list.append(np.zeros(3))
    r,t = UniformRotation_Translation(translation_interval=5.0)
    # return (r @ np.stack(bound_alphaC_loc_clean_list, axis=0).T).T + \
    #     np.repeat(t,len(bound_alphaC_loc_clean_list),axis=0)
    return np.stack(bound_alphaC_loc_clean_list, axis=0)


def filtered_residues_2_list(f_r):
    r_list = torch.tensor([residue_type_pipr(f_r[j][0][2]) for j in range(len(f_r))]).mean(0)
    return r_list

def single_complex(pdb_file_name):
    residue_list_all, chain_num = get_residues(pdb_file_name)
    residues_filtered_list_all = []
    ture_chain_name = []
    all_chains_rep = []
    #------------------------main 1------------------------------
    for i in range(chain_num):
        filtered_residue = filter_residues(residue_list_all[i])
        if filtered_residue != [] and len(filtered_residue) > 1:
            residues_filtered_list_all.append(filtered_residue)
            ture_chain_name.append(filtered_residue[0][0][0])
    
    #------------------------main 1------------------------------
    chain_actual = len(residues_filtered_list_all)
    


    if chain_actual > 3:
        residue_node_loc_list_all = []

        #------------------------main 2------------------------------
        for residues_filtered in residues_filtered_list_all:
            residue_node_loc_list_all.append([get_alphaC_loc_array(residues_filtered)])
        #------------------------main 2------------------------------
        
        for i in range(chain_actual):
            chain_single = filtered_residues_2_list(residues_filtered_list_all[i])
            
            all_chains_rep.append(chain_single)
        
        x = torch.stack(all_chains_rep)
        pairwised=torch.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=-1)
        
        if torch.where(pairwised > 0.9999)[0].shape[0] >= chain_actual * chain_actual:
            all_chains_rep, residue_node_loc_list_all, ture_chain_name = None,None,None
        # if chain_actual <= 10 and torch.where(pairwised > 0.99)[0].shape[0] >= chain_actual + 19:
        #     all_chains_rep, residue_node_loc_list_all, ture_chain_name = None,None,None


        return all_chains_rep, residue_node_loc_list_all, ture_chain_name
    else:
        return None,None,None
def get_protein_info(pdb_file_path):  
    pdb_files = []
    # pdb_files = pdb_files[1000:3000]
    all_chains_rep_list = []
    coor_gt_list = []
    ture_chain_name_list = []
    all_chains_rep, residue_node_loc_list_all, ture_chain_name = single_complex(pdb_file_path)
    return all_chains_rep, residue_node_loc_list_all, ture_chain_name

def process_emb_main(pdb_file_path):
    
    all_chains_rep, residue_node_loc_list_all, ture_chain_name = get_protein_info(pdb_file_path)
    torch.save(all_chains_rep,'multimer_info/chain_rep.pt')
    torch.save(ture_chain_name,'multimer_info/true_chain_name.pt')
    torch.save(residue_node_loc_list_all,'multimer_info/coor_gt.pt')


