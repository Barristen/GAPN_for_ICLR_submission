import torch
from utils_main import *
import os
import torch.nn as nn
# from protein_gnn import *
import scipy.spatial as spa
import itertools
class Env:

    def __init__(self,device):

        # self.complex_all = torch.load('debug_train_dgl_complete.pt')
        self.chain_name_index = ['X']
        self.complex_all_list = []
        self.num_complex_list = []
        self.coor_ture_list = []
        self.chain_name_list = []
        self.pdb_name_list = []
        ii = 0
        for inter in self.chain_name_index:
            self.complex_all_list.append(torch.load('multimer_info/chain_rep.pt'))
            self.num_complex_list.append(len(self.complex_all_list[ii]))
            self.coor_ture_list.append(torch.load('multimer_info/coor_gt.pt'))
            self.chain_name_list.append(torch.load('multimer_info/true_chain_name.pt'))
            # self.pdb_name_list.append(
            #     np.loadtxt('env_data/tmp_true_train_pdb' + '_' + inter + '_pro' + '.txt', dtype=np.str))
            ii += 1

        self.device = device

        aaa = 1




    def reset(self,inter1,round_):
        # set_one = 0
        self.complex_all_list = []
        for inter in self.chain_name_index:
            self.complex_all_list.append(torch.load('multimer_info/chain_rep.pt'))
            self.num_complex_list.append(len(self.complex_all_list[0]))
            self.coor_ture_list.append(torch.load('multimer_info/coor_gt.pt'))
            self.chain_name_list.append(torch.load('multimer_info/true_chain_name.pt'))
        self.current_type_name = self.chain_name_index
        self.complex_all = self.complex_all_list
        self.num_complex = len(self.complex_all)

        self.coor_ture = self.coor_ture_list

        self.chain_name = self.chain_name_list

        # self.pdb_name = self.pdb_name_list[set_chain_type]

        set_one = 0
        self.current_complex = self.complex_all[set_one]
        self.complex_len = len(self.complex_all)

        complex = self.current_complex
        list_com = complex[0].clone().unsqueeze(0)
        ii = 0
        for inter in complex:
            if ii != 0:
                list_com = torch.cat([list_com, inter.unsqueeze(0)], 0)
            ii += 1
        self.current_complex=list_com

        self.curent_pdb_name=1
        self.current_coor_ture=self.coor_ture[set_one]

        self.current_chain_name = self.chain_name[set_one]
        self.current_chain_index=list(np.arange(len(self.current_chain_name)))

        self.current_coor_true=self.coor_ture[set_one]
        # self.current_pdb_name=self.pdb_name[set_one]

        self.list_complex = self.complex_all[set_one]

        # first_sample=np.random.randint(0, len(self.list_complex), 1)[0]
        first_sample = round_
        self.current_state=[self.list_complex[first_sample]]
        # print('len:',len(self.list_complex))
        c_s=self.current_state.copy()
        global_state=self.current_complex
        current_chain_index=self.current_chain_index[first_sample]

        self.list_complex.pop(first_sample)
        l_c=self.list_complex.copy()
        self.current_chain_index.pop(first_sample)

        self.select_chain_index=[]
        self.select_chain_index.append(current_chain_index)
        self.docking_path = []
        self.reward=0
        self.end=False
        return c_s,global_state,current_chain_index,l_c


    def next_step(self,pos,action_prob,rmsd_see=False,clash_done=False):

        if clash_done:
            flat_arr = action_prob.ravel()

            # 获取最大的前十个值的序号
            sorted_indices = np.argsort(flat_arr)[-20:][::-1]

            # 将一维序号转为二维序号
            row_indices, col_indices = np.divmod(sorted_indices, action_prob.shape[1])
            crush_check_l = []
            for id in range(len(row_indices)):
                docked, un_docked = row_indices[id], col_indices[id]
                dock_path = [self.select_chain_index[docked], self.current_chain_index[un_docked]]
                tmp_docking_path_revised = torch.tensor(np.array(self.docking_path + [dock_path]).T)
                crush_check = self.crush_check(self.curent_pdb_name, tmp_docking_path_revised, self.current_chain_name,
                                               self.current_coor_ture)
                crush_check_l.append(crush_check)
            sorted_indices = np.argsort(np.array(crush_check_l))
            ranks_clash = np.argsort(sorted_indices)
            rank_raw = np.arange(len(row_indices))
            rank_final = rank_raw + ranks_clash
            min_index = rank_final.argmin()
            docked, un_docked = row_indices[min_index], col_indices[min_index]
            dock_path = [self.select_chain_index[docked], self.current_chain_index[un_docked]]

        else:
            docked, un_docked=pos
            dock_path=[self.select_chain_index[docked],self.current_chain_index[un_docked]]

            tmp_docking_path_revised = torch.tensor(np.array(self.docking_path + [dock_path]).T)
            crush_check = self.crush_check(self.curent_pdb_name, tmp_docking_path_revised, self.current_chain_name,
                                           self.current_coor_ture)
            if crush_check == True:
                ac_prob = action_prob.copy()
                ac_prob[docked, un_docked] = -10000
                docked, un_docked = np.where(ac_prob == np.max(ac_prob))[0][0], np.where(ac_prob == np.max(ac_prob))[1][0]
                dock_path = [self.select_chain_index[docked], self.current_chain_index[un_docked]]



        self.select_chain_index.append(self.current_chain_index[un_docked])
        self.current_chain_index.pop(un_docked)
        self.docking_path.append(dock_path)
        self.current_state.append(self.list_complex[un_docked])
        c_s = self.current_state.copy()
        self.list_complex.pop(un_docked)
        l_c = self.list_complex.copy()
        docking_path_revised=torch.tensor(np.array(self.docking_path).T)
        rmsd,tm_score,clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list=self.assemble_rmsd_for_inference(self.curent_pdb_name,docking_path_revised,
                                                       self.current_chain_name,self.current_coor_ture)
        rmsd=-rmsd.cpu().detach().numpy()
        # rmsd, tm_score =np.zeros((1)),np.zeros((1))
        # current_reward=rmsd/(self.complex_len*1.0)-self.reward
        current_reward = rmsd  - self.reward
        self.reward=rmsd
        if len(self.list_complex)==0:
            self.end=True
            rmsd, tm_score,clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list = self.assemble_rmsd_for_inference(self.curent_pdb_name, docking_path_revised,
                                                              self.current_chain_name, self.current_coor_ture,self.end,rmsd_see)
        return current_reward,c_s,l_c,self.end,tm_score,rmsd,clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list
   ##chain_list, complex_coor_gt 都是完整的，docking_path D-F-K docking path[[]] 2*2  [[3,11]].T
    # def assemble_rmsd_for_inference(self,pdb_id, docking_path, chain_list, complex_coor_gt):
    #     docking_path = docking_path.long()
    #     unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0), docking_path[1, :].unsqueeze(0)),
    #                                     dim=1).squeeze(0)
    #     all_coor_list = []
    #     root_path = './equidock_result_train/' + pdb_id
    #     full_path_1 = root_path + '_' + chain_list[docking_path[:, 0][0]] + '_' + chain_list[
    #         docking_path[:, 0][1]] + '.npy'
    #     full_path_2 = root_path + '_' + chain_list[docking_path[:, 0][1]] + '_' + chain_list[
    #         docking_path[:, 0][0]] + '.npy'
    #     if os.path.isfile(full_path_1):
    #         coor_complex = np.load(full_path_1, allow_pickle=True)
    #         chain_1, chain_2 = coor_complex[0], coor_complex[1]
    #     else:
    #         coor_complex = np.load(full_path_2, allow_pickle=True)
    #         chain_1, chain_2 = coor_complex[1], coor_complex[0]
    #     all_coor_list.append(chain_1)
    #     all_coor_list.append(chain_2)
    #     for i in range(docking_path.size(1) - 1):
    #         # new_chain_path = docking_path[:,i+1][1]
    #         # exist_chain_path = docking_path[:,i+1][0]
    #         full_path_1 = root_path + '_' + chain_list[docking_path[:, i + 1][0]] + '_' + chain_list[
    #             docking_path[:, i + 1][1]] + '.npy'
    #         full_path_2 = root_path + '_' + chain_list[docking_path[:, i + 1][1]] + '_' + chain_list[
    #             docking_path[:, i + 1][0]] + '.npy'
    #         if os.path.isfile(full_path_1):
    #             coor_complex = np.load(full_path_1, allow_pickle=True)
    #             new_chain_coor, exist_chain_coor = coor_complex[1], coor_complex[0]
    #         else:
    #             coor_complex = np.load(full_path_2, allow_pickle=True)
    #             new_chain_coor, exist_chain_coor = coor_complex[0], coor_complex[1]
    #         exist_docked_chain_coor = all_coor_list[torch.where(unique_docking_path == docking_path[:, i + 1][0])[0]]
    #         r, t = find_rigid_alignment(torch.tensor(exist_chain_coor), torch.tensor(exist_docked_chain_coor))
    #         new_chain_coor_trans = (r @ new_chain_coor.T).T + t
    #         all_coor_list.append(new_chain_coor_trans)
    #
    #     all_coor_list_gt = np.array(complex_coor_gt)[unique_docking_path]
    #     all_coor_gt = all_coor_list_gt[0][0]
    #     all_coor_pred = all_coor_list[0]
    #     for ii in range(all_coor_list_gt.shape[0] - 1):
    #         all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii + 1][0]), axis=0)
    #         all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii + 1]), axis=0)
    #     R0, T0 = find_rigid_alignment(torch.tensor(all_coor_gt), torch.tensor(all_coor_pred))
    #     rmsd_inference = torch.sqrt(
    #         (((R0.mm(torch.tensor(all_coor_gt).T)).T + T0 - torch.tensor(all_coor_pred)) ** 2).sum(axis=1).mean())
    #     N_res = all_coor_gt.shape[0]
    #     eta = 1e-1
    #     d0 = 1.24 * pow((N_res - 15), 1 / 3) - 1.8
    #     R0_refine = torch.tensor(R0, requires_grad=True)
    #     T0_refine = torch.tensor(T0, requires_grad=True)
    #     for iter in range(1000):
    #         tm_1 = R0_refine.mm(torch.tensor(all_coor_gt).T).T + T0_refine
    #         tm_2 = torch.tensor(all_coor_pred)
    #         pdist = nn.PairwiseDistance(p=2)
    #         tmscore_loss = -(1 / ((pdist(tm_1, tm_2) / d0) ** 2 + 1)).mean()
    #         tmscore_loss.backward()
    #         R0_refine = R0_refine - eta * R0_refine.grad.detach()
    #         R0_refine = torch.tensor(R0_refine, requires_grad=True)
    #         T0_refine = T0_refine - eta * T0_refine.grad.detach()
    #         T0_refine = torch.tensor(T0_refine, requires_grad=True)
    #     return rmsd_inference, -tmscore_loss  ###除以链的数量

    def assemble_rmsd_for_inference(self,pdb_id, docking_path, chain_list, complex_coor_gt,end=False,output_tmscore=False):
        r_list = []
        t_list = []
        docking_path = docking_path.long()
        unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0), docking_path[1, :].unsqueeze(0)),
                                        dim=1).squeeze(0)
        all_coor_list = []
        # root_path = '/apdcephfs/share_1364275/kaithgao/RLNDOCK/equidock_process/train/' + pdb_id
        root_path ='./dimer_set/'
        full_path_1 = root_path  + chain_list[docking_path[:, 0][0]] + '_' + chain_list[
            docking_path[:, 0][1]] + '.npy'
        full_path_2 = root_path  + chain_list[docking_path[:, 0][1]] + '_' + chain_list[
            docking_path[:, 0][0]] + '.npy'
        if os.path.isfile(full_path_1):
            coor_complex = np.load(full_path_1, allow_pickle=True)
            chain_1, chain_2 = coor_complex[0], coor_complex[1]
        else:
            coor_complex = np.load(full_path_2, allow_pickle=True)
            chain_1, chain_2 = coor_complex[1], coor_complex[0]
        all_coor_list.append(chain_1)
        all_coor_list.append(chain_2)
        for i in range(docking_path.size(1) - 1):
            # new_chain_path = docking_path[:,i+1][1]
            # exist_chain_path = docking_path[:,i+1][0]
            full_path_1 = root_path  + chain_list[docking_path[:, i + 1][0]] + '_' + chain_list[
                docking_path[:, i + 1][1]] + '.npy'
            full_path_2 = root_path  + chain_list[docking_path[:, i + 1][1]] + '_' + chain_list[
                docking_path[:, i + 1][0]] + '.npy'
            if os.path.isfile(full_path_1):
                coor_complex = np.load(full_path_1, allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[1], coor_complex[0]
            else:
                coor_complex = np.load(full_path_2, allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[0], coor_complex[1]
            exist_docked_chain_coor = all_coor_list[torch.where(unique_docking_path == docking_path[:, i + 1][0])[0]]
            r, t = find_rigid_alignment(torch.tensor(exist_chain_coor), torch.tensor(exist_docked_chain_coor))
            r_list.append(r)
            t_list.append(t)
            new_chain_coor_trans = (r @ new_chain_coor.T).T + t
            all_coor_list.append(new_chain_coor_trans)

        all_coor_list_gt = np.array(complex_coor_gt)[unique_docking_path]
        all_coor_gt = all_coor_list_gt[0][0]
        all_coor_pred = all_coor_list[0]
        for ii in range(all_coor_list_gt.shape[0] - 1):
            all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii + 1][0]), axis=0)
            all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii + 1]), axis=0)
        R0, T0 = find_rigid_alignment(torch.tensor(all_coor_gt).to(self.device), torch.tensor(all_coor_pred).to(self.device))
        rmsd_inference = torch.sqrt(
            (((R0.mm(torch.tensor(all_coor_gt).to(self.device).T)).T + T0 - torch.tensor(all_coor_pred).to(self.device)) ** 2).sum(axis=1).mean())
        if end==True:
            comb_coor = list(itertools.combinations(all_coor_list, 2))
            clash_sum = 0
            for comb in comb_coor:
                # clash_sum+=spa.distance.cdist(comb[0], comb[1]).mean()
                clash_sum += np.sum(spa.distance.cdist(comb[0], comb[1]) < 1)
            if output_tmscore:
                N_res = all_coor_gt.shape[0]
                eta = 1e-4
                d0 = 1.24 * pow((N_res - 15), 1 / 3) - 1.8
                R0_refine = torch.tensor(R0, requires_grad=True).to(self.device)
                T0_refine = torch.tensor(T0, requires_grad=True).to(self.device)
                for iter in range(10000):
                    tm_1 = R0_refine.mm(torch.tensor(all_coor_gt).to(self.device).T).T + T0_refine.to(self.device)
                    tm_2 = torch.tensor(all_coor_pred).to(self.device)
                    pdist = nn.PairwiseDistance(p=2)
                    tmscore_loss = -(1 / ((pdist(tm_1, tm_2) / d0) ** 2 + 1)).mean()
                    tmscore_loss.backward()
                    R0_refine = R0_refine - eta * R0_refine.grad.detach()
                    R0_refine = torch.tensor(R0_refine, requires_grad=True).to(self.device)
                    T0_refine = T0_refine - eta * T0_refine.grad.detach()
                    T0_refine = torch.tensor(T0_refine, requires_grad=True).to(self.device)
            else:
                tmscore_loss = 0

        else:
            tmscore_loss=0
            clash_sum = 0
        return rmsd_inference, -tmscore_loss,clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list

    def crush_check(self, pdb_id, docking_path, chain_list, complex_coor_gt, end=False):
        docking_path = docking_path.long()
        unique_docking_path = torch.cat(
            (docking_path[0][0].unsqueeze(0).unsqueeze(0), docking_path[1, :].unsqueeze(0)),
            dim=1).squeeze(0)
        all_coor_list = []
        # root_path = '/apdcephfs/share_1364275/kaithgao/RLNDOCK/equidock_process/train/' + pdb_id
        root_path = './dimer_set/'
        full_path_1 = root_path  + chain_list[docking_path[:, 0][0]] + '_' + chain_list[
            docking_path[:, 0][1]] + '.npy'
        full_path_2 = root_path  + chain_list[docking_path[:, 0][1]] + '_' + chain_list[
            docking_path[:, 0][0]] + '.npy'
        if os.path.isfile(full_path_1):
            coor_complex = np.load(full_path_1, allow_pickle=True)
            chain_1, chain_2 = coor_complex[0], coor_complex[1]
        else:
            coor_complex = np.load(full_path_2, allow_pickle=True)
            chain_1, chain_2 = coor_complex[1], coor_complex[0]
        all_coor_list.append(chain_1)
        all_coor_list.append(chain_2)
        for i in range(docking_path.size(1) - 1):
            # new_chain_path = docking_path[:,i+1][1]
            # exist_chain_path = docking_path[:,i+1][0]
            full_path_1 = root_path  + chain_list[docking_path[:, i + 1][0]] + '_' + chain_list[
                docking_path[:, i + 1][1]] + '.npy'
            full_path_2 = root_path  + chain_list[docking_path[:, i + 1][1]] + '_' + chain_list[
                docking_path[:, i + 1][0]] + '.npy'
            if os.path.isfile(full_path_1):
                coor_complex = np.load(full_path_1, allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[1], coor_complex[0]
            else:
                coor_complex = np.load(full_path_2, allow_pickle=True)
                new_chain_coor, exist_chain_coor = coor_complex[0], coor_complex[1]
            exist_docked_chain_coor = all_coor_list[
                torch.where(unique_docking_path == docking_path[:, i + 1][0])[0]]
            r, t = find_rigid_alignment(torch.tensor(exist_chain_coor), torch.tensor(exist_docked_chain_coor))
            new_chain_coor_trans = (r @ new_chain_coor.T).T + t
            all_coor_list.append(new_chain_coor_trans)

        all_coor_list_gt = np.array(complex_coor_gt)[unique_docking_path]
        all_coor_gt = all_coor_list_gt[0][0]
        all_coor_pred = all_coor_list[0]
        for ii in range(all_coor_list_gt.shape[0] - 1):
            all_coor_gt = np.concatenate((all_coor_gt, all_coor_list_gt[ii + 1][0]), axis=0)
            all_coor_pred = np.concatenate((all_coor_pred, all_coor_list[ii + 1]), axis=0)

        crash_ratio_l = []
        new_docked_coor = all_coor_list[-1]
        for inter in range(len(all_coor_list) - 1):
            x_2 = all_coor_list[inter]
            ligand_receptor_distance = spa.distance.cdist(new_docked_coor, x_2)
            crash_ratio = np.where(ligand_receptor_distance < 1.5)[0].shape[0] / (
                    new_docked_coor.shape[0] * x_2.shape[0])
            crash_ratio_l.append(crash_ratio)
        crash_info = np.array(crash_ratio_l).mean()

        # for inter in crash_ratio_l:
        #     if inter > 0.01:
        #         crash_info = True
        #         break
        return crash_info