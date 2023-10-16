import torch
import itertools
import scipy.spatial as spa
from tqdm import tqdm
from biopandas.pdb import PandasPdb
import numpy as np
import warnings
warnings.filterwarnings("ignore")
def preprocess_unbound_bound(bound_ligand_residues, bound_receptor_residues, graph_nodes, pos_cutoff=8.0, inference=False):
    #######################
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
   ##########################

    bound_predic_ligand_filtered = filter_residues(bound_ligand_residues)
    unbound_predic_ligand_filtered = bound_predic_ligand_filtered

    bound_predic_receptor_filtered = filter_residues(bound_receptor_residues)
    unbound_predic_receptor_filtered = bound_predic_receptor_filtered

    bound_predic_ligand_clean_list = bound_predic_ligand_filtered
    unbound_predic_ligand_clean_list = unbound_predic_ligand_filtered

    bound_predic_receptor_clean_list= bound_predic_receptor_filtered
    unbound_predic_receptor_clean_list = unbound_predic_receptor_filtered

    ###################
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
        return np.stack(bound_alphaC_loc_clean_list, axis=0)  # (N_res,3)

   ####################

    assert graph_nodes == 'residues'
    bound_receptor_repres_nodes_loc_array = get_alphaC_loc_array(bound_predic_receptor_clean_list)
    bound_ligand_repres_nodes_loc_array = get_alphaC_loc_array(bound_predic_ligand_clean_list)

    return unbound_predic_ligand_clean_list, unbound_predic_receptor_clean_list, \
           bound_ligand_repres_nodes_loc_array, bound_receptor_repres_nodes_loc_array

def find_rigid_alignment(A, B):
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()

def get_rot_mat(euler_angles):
    roll = euler_angles[0]
    yaw = euler_angles[1]
    pitch = euler_angles[2]

    tensor_0 = torch.zeros([])
    tensor_1 = torch.ones([])
    cos = torch.cos
    sin = torch.sin

    RX = torch.stack([
        torch.stack([tensor_1, tensor_0, tensor_0]),
        torch.stack([tensor_0, cos(roll), -sin(roll)]),
        torch.stack([tensor_0, sin(roll), cos(roll)])]).reshape(3, 3)

    RY = torch.stack([
        torch.stack([cos(pitch), tensor_0, sin(pitch)]),
        torch.stack([tensor_0, tensor_1, tensor_0]),
        torch.stack([-sin(pitch), tensor_0, cos(pitch)])]).reshape(3, 3)

    RZ = torch.stack([
        torch.stack([cos(yaw), -sin(yaw), tensor_0]),
        torch.stack([sin(yaw), cos(yaw), tensor_0]),
        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    return R



def get_residues(df):
    drop_list = []
    res_name_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR',
           'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ',
           'TYS', 'CYX', 'GLZ']
    drop_list = list(set([item for item in df['residue_name'] if item not in set(res_name_list)]))
    for ele in drop_list:
        df = df.drop(index=df[df['residue_name']==ele].index)
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    residues = list(df.groupby(['chain', 'residue', 'resname']))  ## Not the same as sequence order !
    return residues

def get_nodes_coors_numpy(filename, all_atoms=False):
            df = PandasPdb().read_pdb(filename).df['ATOM']
            if not all_atoms:
                return torch.from_numpy(df[df['atom_name'] == 'CA'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))
            return torch.from_numpy(df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32))

def dimer_main(pdb_path):
    chain_name = torch.load('multimer_info/true_chain_name.pt')

    output_dir = 'dimer_set/'

    ppdb = PandasPdb().read_pdb(pdb_path)

    df = ppdb.df['ATOM']
    all_combine = list(itertools.combinations(chain_name,2))
    for sing_com in tqdm(all_combine):
        chain_1_df_atom = df[df['chain_id'].isin([sing_com[0]])]
        chain_2_df_atom = df[df['chain_id'].isin([sing_com[1]])]
        

        unbound_predic_ligand, \
        unbound_predic_receptor, \
        bound_ligand_repres_nodes_loc_clean_array,\
        bound_receptor_repres_nodes_loc_clean_array = preprocess_unbound_bound(
            get_residues(chain_1_df_atom), get_residues(chain_2_df_atom),
            graph_nodes='residues', pos_cutoff=8, inference=True)

        ligand_receptor_distance = spa.distance.cdist(bound_ligand_repres_nodes_loc_clean_array, bound_receptor_repres_nodes_loc_clean_array)

        if np.where(ligand_receptor_distance < 8)[0].shape[0] < ligand_receptor_distance.shape[0] * ligand_receptor_distance.shape[1] * 0.01 * 0.01:
            protein_1_ca_coor = bound_ligand_repres_nodes_loc_clean_array
            protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array            

            interface_num = int(0.05 * min(protein_2_ca_coor.shape[0],protein_1_ca_coor.shape[0])) + 2

            r,t = find_rigid_alignment(torch.tensor(protein_1_ca_coor[:interface_num,:]),torch.tensor(protein_2_ca_coor[:interface_num,:]))
            
            protein_1_ca_coor = ((r @ bound_ligand_repres_nodes_loc_clean_array.T).T + t).numpy()
            protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array
        else:
            protein_1_ca_coor = bound_ligand_repres_nodes_loc_clean_array
            protein_2_ca_coor = bound_receptor_repres_nodes_loc_clean_array

            

        coor_pair_list = []
        coor_pair_list.append(protein_1_ca_coor)
        coor_pair_list.append(protein_2_ca_coor)
        final_pair_name = output_dir + sing_com[0] + '_' + sing_com[1] + '.npy'
        np.save(final_pair_name,np.array(coor_pair_list))


