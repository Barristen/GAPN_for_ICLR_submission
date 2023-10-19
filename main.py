import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from env_test import Env
from model import actor,critic
import torch
# import dgl
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from process_to_emb import *
from collections import deque
from torch.distributions import Categorical
from dimer_code.utils import *
from dimer_code.dimer_main import *
from biopandas.pdb import PandasPdb
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import py3Dmol
import subprocess
import nbformat
from nbconvert import HTMLExporter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(1)

def run_and_display_notebook(name):
    st.write("Running notebook...")
    try:
        # Run the notebook
        subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", name])
        
        # Read the executed notebook
        with open(name + ".nbconvert.ipynb", "r") as notebook_file:
            notebook_content = nbformat.read(notebook_file, as_version=4)
        
        output_cells = []
        for cell in notebook_content.cells:
            if cell.cell_type == "code":
                output_cell = copy.deepcopy(cell)  # Create a deep copy so as not to modify the original
                output_cell.source = ""  # Remove the source code
                output_cells.append(output_cell)
            else:
                output_cells.append(cell)
        notebook_content.cells = output_cells
        # notebook_content.cells = [cell for cell in notebook_content.cells if cell.cell_type != "code"]
        # Create a new list to store the cells

        # Export the executed notebook as HTML
        html_exporter = HTMLExporter()
        html_output, _ = html_exporter.from_notebook_node(notebook_content)
        
        # Display the HTML content in Streamlit
        st.components.v1.html(html_output, width=800, height=800)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
class run_docking:
    def __init__(self):
        ## env
        self.env=Env(device)
        ## model
        self.learning_rate_a=3e-3
        self.learning_rate_c = 6e-3
        self.ac = actor(13, 32).to(device)
        self.ac.load_state_dict(torch.load("model/ac_chain_16_22_full.pkl",map_location=torch.device('cpu')))

        ## para
        self.inter_num=1
        self.buffer = deque(maxlen=104)
        self.batch_size=100
        self.gamma=1
        self.clip_param=0.2
        self.en_para=0.005
        self.max_grad_norm=0.5
        # self.action_type="algo"   #"random"
        # self.action_type = "random"
        self.action_type = "prob_greedy"

    def main(self):
        rmsd_l=[]
        tm_score_l=[]
        clash_num_l=[]
        chain_name_index = [3, 4, 5, 6, 7, 8, 9, 10]
        for inter in range(self.inter_num):
            chain_shang=inter//20
            # range_len=int(chain_name_index[chain_shang])
            tm_score_max=0
            current_state, global_state, current_chain_name, undocking_state = self.env.reset(inter, 0)
            for ii in range(len(global_state)):
                end=False
                current_state, global_state, current_chain_name, undocking_state = self.env.reset(inter,ii)
                while not end:
                    if self.action_type=="random":
                        action_prob, action = self.next_action_random(current_state, undocking_state, global_state)

                    elif self.action_type=="prob_greedy":
                        action_prob, action = self.next_action_greedy(current_state, undocking_state, global_state)
                    else:
                        action_prob,action=self.next_action(current_state,undocking_state,global_state)
                    row,col=action_prob.shape
                    shang=action[0]//col
                    yu=action[0]%col
                    reward,next_state,next_undocking_state,end,tm_score,rmsd,clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list =self.env.next_step((shang,yu),action_prob)
                    if end==True:
                        aaa=1
                    current_state=next_state
                    undocking_state=next_undocking_state
                # print(tm_score,rmsd,clash_sum)

                clash_num_l.append(clash_sum)
                # print(tm_score_tem,rmsd_tem,clash_sum)
                # if tm_score_tem>tm_score_max:
                #     rmsd_min=rmsd_tem
                #     tm_score_max=tm_score_tem

            # index_=self.sample_indices(np.array(clash_num_l), size=1)[0]
            index_=np.argmin(np.array(clash_num_l))
            end = False
            current_state, global_state, current_chain_name, undocking_state = self.env.reset(inter, index_)
            while not end:
                if self.action_type == "random":
                    action_prob, action = self.next_action_random(current_state, undocking_state, global_state)

                elif self.action_type == "prob_greedy":
                    action_prob, action = self.next_action_greedy(current_state, undocking_state, global_state)
                else:
                    action_prob, action = self.next_action(current_state, undocking_state, global_state)
                row, col = action_prob.shape
                shang = action[0] // col
                yu = action[0] % col
                reward, next_state, next_undocking_state, end, tm_score, rmsd, clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list = self.env.next_step(
                    (shang, yu), action_prob,True,True)
                current_state = next_state
                undocking_state = next_undocking_state

            tm_score = tm_score.cpu().detach().numpy()
            rmsd = rmsd.cpu().detach().numpy()

            print("rmsd:",rmsd,"tm_score",tm_score,"docking_path:",docking_path)
        # avg_rmsd=np.array(rmsd_l).mean()
        # avg_tm_score=np.array(tm_score_l).mean()
        # rmsd_l_array=np.array(rmsd_l).reshape(-1,20).mean(axis=-1)
        # tm_score_l_array=np.array(tm_score_l).reshape(-1,20).mean(axis=-1)
        # print("avg_rmsd",avg_rmsd,"avg_tm_score",avg_tm_score)
        # print("avg_rmsd_group",rmsd_l_array,"avg_tm_score_group",tm_score_l_array)
        return docking_path,all_coor_list_gt,all_coor_list,r_list,t_list

    def sample_indices(self,arr, size=1):
        # 反转数组值作为概率的基础
        inverse_values = 1 / (arr + 1)  # 这里+1是为了避免除以0的情况，根据实际情况调整

        # 归一化概率
        probabilities = inverse_values / np.sum(inverse_values)

        # 按照概率取样索
        return np.random.choice(len(arr), size=size, p=probabilities)

    def next_action(self,current_state,undocking_state,global_state):
        with torch.no_grad():
            action_prob,action = self.ac(self.batch(current_state).to(device), self.batch(undocking_state).to(device),global_state.to(device))
        return  action_prob.cpu().numpy(),action.cpu().numpy()

    def next_action_greedy(self,current_state,undocking_state,global_state):
        with torch.no_grad():
            action_prob,action = self.ac(self.batch(current_state).to(device), self.batch(undocking_state).to(device),global_state.to(device))
        action_p=action_prob.clone()
        action_p=action_p.reshape(-1)
        action_p=action_p.cpu().numpy()
        action=action_p.argmax()

        return  action_prob.cpu().numpy(),np.array(action).reshape(1)

    def next_action_random(self,current_state,undocking_state,global_state):
        with torch.no_grad():
            action_prob,action = self.ac(self.batch(current_state).to(device), self.batch(undocking_state).to(device),global_state.to(device))
        all_len=len(action_prob.reshape(-1))
        action=np.random.randint(0, all_len, 1).reshape(1)
        return  action_prob.cpu().numpy(),action

    def batch(self,state):
        list_com = state[0].clone().unsqueeze(0)
        ii = 0
        for inter in state:
            if ii != 0:
                list_com = torch.cat([list_com, inter.unsqueeze(0)], 0)
            ii += 1
        return list_com



if __name__ == "__main__":
    st.title("PDB处理")

    uploaded_file = st.file_uploader("上传PDB文件")

    if uploaded_file is not None:
        st.write("正在处理文件...")
        temp_path = os.path.join("temp.pdb")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        pdb_path = 'temp.pdb'

        st.write('Processing multimer information.....')
        process_emb_main(pdb_path)
        st.write('Done')
        st.write('Preparing all possible dimer structures.....')
        dimer_main(pdb_path)
        st.write('Done')
        run_dock=run_docking()
        docking_path,all_coor_gt,all_coor_pred,r_list,t_list = run_dock.main()
        unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0), docking_path[1, :].unsqueeze(0)),
                                            dim=1).squeeze(0)
        

        ppdb = PandasPdb().read_pdb(pdb_path)
        chain_name = torch.load('multimer_info/true_chain_name.pt')
        df = ppdb.df['ATOM']
        df_copy = df.copy()
        atom_list = []
        atom_first = df[df['chain_id'].isin([chain_name[unique_docking_path[0]]])][['x_coord', 'y_coord', 'z_coord']]
        atom_second = df[df['chain_id'].isin([chain_name[unique_docking_path[1]]])][['x_coord', 'y_coord', 'z_coord']]
        atom_list.append(atom_first)
        atom_list.append(atom_second)

        for i in range(len(chain_name)-2):        
            chain_df_atom = df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])][['x_coord', 'y_coord', 'z_coord']]
            atom_new = (r_list[i] @ chain_df_atom.T).T + t_list[i]
            df.loc[df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])].index,['x_coord']] = np.array(atom_new)[:,0]
            df.loc[df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])].index,['y_coord']] = np.array(atom_new)[:,1]
            df.loc[df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])].index,['z_coord']] = np.array(atom_new)[:,2]
            atom_list.append(atom_new)
        
        pdb_df = PandasPdb().read_pdb(pdb_path)
        pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = df[['x_coord', 'y_coord', 'z_coord']]
        pdb_df.to_pdb(path=pdb_path[:-4]+'_pred.pdb', records=['ATOM'], gz=False)  


        with open('temp_pred.pdb', 'rb') as f:
            file_content = f.read()

        # 提供一个下载按钮
        if st.download_button('Download Processed PDB File', file_content, file_name='processed_pdb_file.pdb'):
            st.write("File is being downloaded!")

        code_string = """
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]='0'
        
        from env_test import Env
        from model import actor,critic
        import torch
        # import dgl
        import random
        import torch.nn as nn
        import numpy as np
        import torch.nn.functional as F
        from process_to_emb import *
        from collections import deque
        from torch.distributions import Categorical
        from dimer_code.utils import *
        from dimer_code.dimer_main import *
        from biopandas.pdb import PandasPdb
        import streamlit as st
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import py3Dmol
        import subprocess
        import nbformat
        from nbconvert import HTMLExporter
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(1)
        
        def run_and_display_notebook(name):
            st.write("Running notebook...")
            try:
                # Run the notebook
                subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", name])
                
                # Read the executed notebook
                with open(name + ".nbconvert.ipynb", "r") as notebook_file:
                    notebook_content = nbformat.read(notebook_file, as_version=4)
                
                # notebook_content.cells = [cell for cell in notebook_content.cells if cell.cell_type != "code"]
                # Create a new list to store the cells
        
                # Export the executed notebook as HTML
                html_exporter = HTMLExporter()
                html_output, _ = html_exporter.from_notebook_node(notebook_content)
                
                # Display the HTML content in Streamlit
                st.components.v1.html(html_output, width=800, height=800)
        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        class run_docking:
            def __init__(self):
                ## env
                self.env=Env(device)
                ## model
                self.learning_rate_a=3e-3
                self.learning_rate_c = 6e-3
                self.ac = actor(13, 32).to(device)
                self.ac.load_state_dict(torch.load("model/ac_chain_16_22_full.pkl",map_location=torch.device('cpu')))
        
                ## para
                self.inter_num=1
                self.buffer = deque(maxlen=104)
                self.batch_size=100
                self.gamma=1
                self.clip_param=0.2
                self.en_para=0.005
                self.max_grad_norm=0.5
                # self.action_type="algo"   #"random"
                # self.action_type = "random"
                self.action_type = "prob_greedy"
        
            def main(self):
                rmsd_l=[]
                tm_score_l=[]
                clash_num_l=[]
                chain_name_index = [3, 4, 5, 6, 7, 8, 9, 10]
                for inter in range(self.inter_num):
                    chain_shang=inter//20
                    # range_len=int(chain_name_index[chain_shang])
                    tm_score_max=0
                    current_state, global_state, current_chain_name, undocking_state = self.env.reset(inter, 0)
                    for ii in range(len(global_state)):
                        end=False
                        current_state, global_state, current_chain_name, undocking_state = self.env.reset(inter,ii)
                        while not end:
                            if self.action_type=="random":
                                action_prob, action = self.next_action_random(current_state, undocking_state, global_state)
        
                            elif self.action_type=="prob_greedy":
                                action_prob, action = self.next_action_greedy(current_state, undocking_state, global_state)
                            else:
                                action_prob,action=self.next_action(current_state,undocking_state,global_state)
                            row,col=action_prob.shape
                            shang=action[0]//col
                            yu=action[0]%col
                            reward,next_state,next_undocking_state,end,tm_score,rmsd,clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list =self.env.next_step((shang,yu),action_prob)
                            if end==True:
                                aaa=1
                            current_state=next_state
                            undocking_state=next_undocking_state
                        # print(tm_score,rmsd,clash_sum)
        
                        clash_num_l.append(clash_sum)
                        # print(tm_score_tem,rmsd_tem,clash_sum)
                        # if tm_score_tem>tm_score_max:
                        #     rmsd_min=rmsd_tem
                        #     tm_score_max=tm_score_tem
        
                    # index_=self.sample_indices(np.array(clash_num_l), size=1)[0]
                    index_=np.argmin(np.array(clash_num_l))
                    end = False
                    current_state, global_state, current_chain_name, undocking_state = self.env.reset(inter, index_)
                    while not end:
                        if self.action_type == "random":
                            action_prob, action = self.next_action_random(current_state, undocking_state, global_state)
        
                        elif self.action_type == "prob_greedy":
                            action_prob, action = self.next_action_greedy(current_state, undocking_state, global_state)
                        else:
                            action_prob, action = self.next_action(current_state, undocking_state, global_state)
                        row, col = action_prob.shape
                        shang = action[0] // col
                        yu = action[0] % col
                        reward, next_state, next_undocking_state, end, tm_score, rmsd, clash_sum,docking_path,all_coor_list_gt,all_coor_list,r_list,t_list = self.env.next_step(
                            (shang, yu), action_prob,True,True)
                        current_state = next_state
                        undocking_state = next_undocking_state
        
                    tm_score = tm_score.cpu().detach().numpy()
                    rmsd = rmsd.cpu().detach().numpy()
        
                    print("rmsd:",rmsd,"tm_score",tm_score,"docking_path:",docking_path)
                # avg_rmsd=np.array(rmsd_l).mean()
                # avg_tm_score=np.array(tm_score_l).mean()
                # rmsd_l_array=np.array(rmsd_l).reshape(-1,20).mean(axis=-1)
                # tm_score_l_array=np.array(tm_score_l).reshape(-1,20).mean(axis=-1)
                # print("avg_rmsd",avg_rmsd,"avg_tm_score",avg_tm_score)
                # print("avg_rmsd_group",rmsd_l_array,"avg_tm_score_group",tm_score_l_array)
                return docking_path,all_coor_list_gt,all_coor_list,r_list,t_list
        
            def sample_indices(self,arr, size=1):
                inverse_values = 1 / (arr + 1)  
                probabilities = inverse_values / np.sum(inverse_values)
                return np.random.choice(len(arr), size=size, p=probabilities)
        
            def next_action(self,current_state,undocking_state,global_state):
                with torch.no_grad():
                    action_prob,action = self.ac(self.batch(current_state).to(device), self.batch(undocking_state).to(device),global_state.to(device))
                return  action_prob.cpu().numpy(),action.cpu().numpy()
        
            def next_action_greedy(self,current_state,undocking_state,global_state):
                with torch.no_grad():
                    action_prob,action = self.ac(self.batch(current_state).to(device), self.batch(undocking_state).to(device),global_state.to(device))
                action_p=action_prob.clone()
                action_p=action_p.reshape(-1)
                action_p=action_p.cpu().numpy()
                action=action_p.argmax()
        
                return  action_prob.cpu().numpy(),np.array(action).reshape(1)
        
            def next_action_random(self,current_state,undocking_state,global_state):
                with torch.no_grad():
                    action_prob,action = self.ac(self.batch(current_state).to(device), self.batch(undocking_state).to(device),global_state.to(device))
                all_len=len(action_prob.reshape(-1))
                action=np.random.randint(0, all_len, 1).reshape(1)
                return  action_prob.cpu().numpy(),action
        
            def batch(self,state):
                list_com = state[0].clone().unsqueeze(0)
                ii = 0
                for inter in state:
                    if ii != 0:
                        list_com = torch.cat([list_com, inter.unsqueeze(0)], 0)
                    ii += 1
                return list_com
        
        
        
        if __name__ == "__main__":
            st.title("PDB处理")
        
            uploaded_file = st.file_uploader("上传PDB文件")
        
            if uploaded_file is not None:
                st.write("正在处理文件...")
                temp_path = os.path.join("temp.pdb")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
        
                pdb_path = 'temp.pdb'
        
                st.write('Processing multimer information.....')
                process_emb_main(pdb_path)
                st.write('Done')
                st.write('Preparing all possible dimer structures.....')
                dimer_main(pdb_path)
                st.write('Done')
                run_dock=run_docking()
                docking_path,all_coor_gt,all_coor_pred,r_list,t_list = run_dock.main()
                unique_docking_path = torch.cat((docking_path[0][0].unsqueeze(0).unsqueeze(0), docking_path[1, :].unsqueeze(0)),
                                                    dim=1).squeeze(0)
                
        
                ppdb = PandasPdb().read_pdb(pdb_path)
                chain_name = torch.load('multimer_info/true_chain_name.pt')
                df = ppdb.df['ATOM']
                df_copy = df.copy()
                atom_list = []
                atom_first = df[df['chain_id'].isin([chain_name[unique_docking_path[0]]])][['x_coord', 'y_coord', 'z_coord']]
                atom_second = df[df['chain_id'].isin([chain_name[unique_docking_path[1]]])][['x_coord', 'y_coord', 'z_coord']]
                atom_list.append(atom_first)
                atom_list.append(atom_second)
        
                for i in range(len(chain_name)-2):        
                    chain_df_atom = df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])][['x_coord', 'y_coord', 'z_coord']]
                    atom_new = (r_list[i] @ chain_df_atom.T).T + t_list[i]
                    df.loc[df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])].index,['x_coord']] = np.array(atom_new)[:,0]
                    df.loc[df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])].index,['y_coord']] = np.array(atom_new)[:,1]
                    df.loc[df[df['chain_id'].isin([chain_name[unique_docking_path[i+2]]])].index,['z_coord']] = np.array(atom_new)[:,2]
                    atom_list.append(atom_new)
                
                pdb_df = PandasPdb().read_pdb(pdb_path)
                pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = df[['x_coord', 'y_coord', 'z_coord']]
                pdb_df.to_pdb(path=pdb_path[:-4]+'_pred.pdb', records=['ATOM'], gz=False)  
                
        """
        
        # 判断用户是否点击了 "show more" 按钮
        if st.button("Show more"):
            # 用户点击后显示完整的代码块
            st.code(code_string, language='python')
        else:
            # 用户没点击前只显示部分代码块（例如前5行）
            st.code("\n".join(code_string.split("\n")[:5]), language='python')

        st.write("原始结构可视化：")
        run_and_display_notebook('show')  

        st.write("预测结构可视化：")
        run_and_display_notebook('show_2')
        
    
