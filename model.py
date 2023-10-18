# from protein_gnn import *
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch
class actor(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats
    ):
        super(actor, self).__init__()
        self.gcn_1=nn.Linear(in_feats,h_feats)
        self.gcn_2 = nn.Linear(in_feats, h_feats)
        self.gcn_3 = nn.Linear(in_feats, h_feats)
        self.layer =nn.Linear(h_feats*2,h_feats)

    def forward(self,p_c,p_u,p_all):
        p_c=self.gcn_1(p_c)
        p_u=self.gcn_2(p_u)
        p_all=self.gcn_3(p_all).mean(0).reshape(1,-1)
        p_all=p_all.repeat(len(p_c),1)
        p_c=torch.cat([p_c,p_all],-1)
        p_c=F.relu(self.layer(p_c))
        action_=torch.mm(p_c,p_u.T)
        row,col=action_.shape
        action_prob=F.softmax(action_.reshape(-1))
        dis = Categorical(action_prob)
        action = dis.sample()
        action=action.reshape(-1)
        action_prob=action_prob.reshape(row, col)
        return action_prob,action


class critic(nn.Module):
    def __init__(
            self,
            in_feats,
            h_feats
    ):
        super(critic, self).__init__()
        self.gcn=nn.Linear(in_feats,h_feats)
        self.hidden_dim = 32
        self.attn = nn.MultiheadAttention(self.hidden_dim, 8)
        self.Q = nn.Linear(h_feats, self.hidden_dim)
        self.V = nn.Linear(h_feats, self.hidden_dim)
        self.K = nn.Linear(h_feats, self.hidden_dim)
        self.layer = nn.Linear(self.hidden_dim+h_feats, 1)
    def forward(self, p_c,p_all):
        p_c = self.gcn(p_c)
        p_all = self.gcn(p_all).mean(0).reshape(1, -1)
        Query = self.Q(p_c)
        Query = F.relu(Query)
        Query = Query.unsqueeze(0)

        Value = self.V(p_c)
        Value = F.relu(Value)
        Value = Value.unsqueeze(0)

        Key = self.K(p_c)
        Key = F.relu(Key)
        Key = Key.unsqueeze(0)
        p_c, _ = self.attn(Query.transpose(0, 1), Key.transpose(0, 1), Value.transpose(0, 1))
        p_c = p_c.squeeze(1)
        p_all = p_all.repeat(len(p_c), 1)
        p_c = torch.cat([p_c, p_all], -1)
        value = F.relu(self.layer(p_c)).mean(0)
        return value

if __name__ == "__main__":
    ### 24   [（1,2，26） (22,26) (2,22)]  [（1,3，26） (21,26) (3,21)]
    complex_all = torch.load('all_chains_rep_list.pt')
    complex=complex_all[0]
    list_com=complex[0].clone().unsqueeze(0)
    ii=0
    for inter in complex:
        if ii!=0:
            list_com=torch.cat([list_com,inter.unsqueeze(0)],0)
        ii+=1
    pc=list_com[:2]
    pu = list_com[2:]
    ac=actor(13,32)
    action=ac(pc,pu,list_com)
    cr=critic(13,32)
    value=cr(pc,list_com)
    aaa=1