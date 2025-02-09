import torch
from torch import nn
import numpy as np
class Doc_embedding(torch.nn.Module):
    def __init__(self,encoder,encode_dim,hidden_size, embedding_talble,device):
        super().__init__()
        self.device = device
        self.encode_dim = encode_dim
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.embedding_table = embedding_talble
        self.doc_transfer_layer = nn.Linear(self.encode_dim, self.hidden_size).to(self.device)

    def forward(self,input_doc_id):
        doc_embedding_batch = []
        for id in input_doc_id:
            doc_embedding_batch.append(self.embedding_table[id].to(self.device))
        doc_embedding_batch = torch.cat(doc_embedding_batch,dim=0).to(self.device)
        output_embeds = self.doc_transfer_layer(doc_embedding_batch)
        return output_embeds

class Feature_embedding(torch.nn.Module):
    def __init__(self,feature_dim,hidden_size, feature_table,device):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.feature_table = feature_table
        self.doc_transfer_layer = nn.Linear(self.feature_dim, self.hidden_size).to(self.device)

    def forward(self,input_feature_id):
        feature_batch = []
        # print('len of input_feature_id in Feature_embedding is:')
        # print(len(input_feature_id))
        for id in input_feature_id:
            feature_batch.append(self.feature_table[id[0]][id[1]]['feature'].unsqueeze(0).to(self.device))
        feature_batch = torch.cat(feature_batch,dim=0).to(self.device) #[len_feature,dim]
        output_embeds = self.doc_transfer_layer(feature_batch) #[len_seq,hidden_size]
        return output_embeds

class Feature_embedding_pre_process(torch.nn.Module):
    def __init__(self,feature_dim,hidden_size, feature_table,device):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.feature_table = feature_table
        self.doc_transfer_layer = nn.Linear(self.feature_dim, self.hidden_size).to(self.device)
        self.RELU_input = nn.ReLU(inplace=False).to(device)


    def forward(self,input_feature_id,is_train=True):
        feature_batch = []
        # print('len of input_feature_id in Feature_embedding is:')
        # print(len(input_feature_id))
        for id in input_feature_id:
            feature = self.feature_table[id[0]][id[1]]['feature'].unsqueeze(0).to(self.device) #[1,feature_dim]
            res = torch.log(torch.abs(feature) + 1).mul(torch.sign(feature))
            # if is_train:
            #     res = res + torch.randn(1,self.feature_dim).to(self.device)
            #FRB_batch.append(self.FRB_2(self.FRB_1(self.FRB_0(res))))
            feature_batch.append(res)
        feature_batch = torch.cat(feature_batch,dim=0).to(self.device) #[len_seq,feature_dim]
        #FRB_batch = self.RELU_out(self.FC_out(self.FRB_2(self.FRB_1(self.FRB_0(feature_batch))))) # [len_seq,feature_dim]----->[len_seq,hidden_size]
        output_embeds = self.RELU_input(self.doc_transfer_layer(feature_batch)) #[len_seq,hidden_size]
        return output_embeds,feature_batch


class Feature_embedding_yahoo(torch.nn.Module):
    def __init__(self,feature_dim,hidden_size, feature_table,device):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.feature_table = feature_table
        self.doc_transfer_layer = nn.Linear(self.feature_dim, self.hidden_size).to(self.device)
        self.RELU_input = nn.ReLU(inplace=False).to(device)


    def forward(self,input_feature_id,is_train=True):
        feature_batch = []
        # print('len of input_feature_id in Feature_embedding is:')
        # print(len(input_feature_id))
        for id in input_feature_id:
            feature = self.feature_table[id[0]][id[1]]['feature'].unsqueeze(0).to(self.device) #[1,feature_dim]
            # = torch.log(torch.abs(feature) + 1).mul(torch.sign(feature))
            # if is_train:
            #     res = res + torch.randn(1,self.feature_dim).to(self.device)
            #FRB_batch.append(self.FRB_2(self.FRB_1(self.FRB_0(res))))
            feature_batch.append(feature)
        feature_batch = torch.cat(feature_batch,dim=0).to(self.device) #[len_seq,feature_dim]
        #FRB_batch = self.RELU_out(self.FC_out(self.FRB_2(self.FRB_1(self.FRB_0(feature_batch))))) # [len_seq,feature_dim]----->[len_seq,hidden_size]
        output_embeds = self.RELU_input(self.doc_transfer_layer(feature_batch)) #[len_seq,hidden_size]
        # print('output')
        # print(output_embeds.shape)
        # print('feature')
        # print(feature_batch.shape)
        return output_embeds,feature_batch



class Feature_embedding_yahoo_pos(torch.nn.Module):
    def __init__(self,feature_dim,hidden_size, feature_table,device):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.feature_table = feature_table
        self.doc_transfer_layer = nn.Linear(self.feature_dim, self.hidden_size).to(self.device)
        self.pos_embedding = nn.Embedding(40,self.feature_dim).to(self.device)
        #self.RELU_input = nn.ReLU(inplace=False).to(device)
        self.RELU_input = nn.SiLU(inplace=False).to(device)

    def forward(self,input_feature_id,is_train=True):
        feature_batch = []
        count = 0
        all = len(input_feature_id)
        frb_embeds = []
        for id in input_feature_id:
            feature_origin = self.feature_table[id[0]][id[1]]['feature'].unsqueeze(0).to(self.device) #[1,feature_dim]
            #feature_origin = self.feature_table[id[0]][id[1]].unsqueeze(0).to(self.device) #[1,feature_dim]
            feature_origin = torch.log(torch.abs(feature_origin) + 1).mul(torch.sign(feature_origin))
            # if is_train:
            #     res = res + torch.randn(1,self.feature_dim).to(self.device)
            #FRB_batch.append(self.FRB_2(self.FRB_1(self.FRB_0(res))))
            #feature = torch.cat([feature_origin, ((all-count) / all)*torch.ones(1,701).to(self.device)],dim=1)
            pos = np.array([count])
            pos = torch.from_numpy(pos).to(self.device)
            pos_emb = self.pos_embedding(pos)
            feature = torch.add(feature_origin,pos_emb)
            feature_batch.append(feature)
            frb_embeds.append(feature_origin)
            count += 1
        frb_embeds = torch.cat(frb_embeds,dim=0).to(self.device) #[len_seq,feature_dim]
        feature_batch = torch.cat(feature_batch, dim=0).to(self.device)
        #FRB_batch = self.RELU_out(self.FC_out(self.FRB_2(self.FRB_1(self.FRB_0(feature_batch))))) # [len_seq,feature_dim]----->[len_seq,hidden_size]
        output_embeds = self.RELU_input(self.doc_transfer_layer(feature_batch)) #[len_seq,hidden_size]
        # print('output')
        # print(output_embeds.shape)
        # print('feature')
        # print(feature_batch.shape)
        return output_embeds,frb_embeds

class Feature_embedding_pre_process_pos(torch.nn.Module):
    def __init__(self,feature_dim,hidden_size, feature_table,device):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.feature_table = feature_table
        self.doc_transfer_layer = nn.Linear(self.feature_dim, self.hidden_size).to(self.device)
        self.RELU_input = nn.ReLU(inplace=False).to(device)
        #self.add_pos_embedding = nn.Linear(2*self.feature_dim, self.hidden_size).to(self.device)

    def forward(self,input_feature_id,is_train=True):
        feature_batch = []
        # print('len of input_feature_id in Feature_embedding is:')
        # print(len(input_feature_id))
        pos_batch = []
        count = 0
        for id in input_feature_id:
            feature = self.feature_table[id[0]][id[1]]['feature'].unsqueeze(0).to(self.device) #[1,feature_dim]
            #feature = self.feature_table[id[0]][id[1]].unsqueeze(0).to(self.device)  # [1,feature_dim]
            res = torch.log(torch.abs(feature) + 1).mul(torch.sign(feature))
            # if is_train:
            #     res = res + torch.randn(1,self.feature_dim).to(self.device)
            #FRB_batch.append(self.FRB_2(self.FRB_1(self.FRB_0(res))))
            count += 1
            #pos_batch.append(count*torch.ones(1,self.hidden_size))
            feature_batch.append(res)
        feature_batch = torch.cat(feature_batch,dim=0).to(self.device) #[len_seq,feature_dim]
        #pos_batch = torch.cat(pos_batch, dim=0).to(self.device)
        output_embeds = self.RELU_input(self.doc_transfer_layer(feature_batch)) #[len_seq,hidden_size]
        #output_embeds = output_embeds + pos_batch
        return output_embeds,feature_batch

class FC_RE_BN(torch.nn.Module):
    def __init__(self,feature_dim,device):
        super().__init__()
        self.FFN = nn.Linear(feature_dim, feature_dim).to(device)
        self.RELU = nn.ReLU(inplace=False).to(device)
        self.BN = nn.BatchNorm1d(feature_dim,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True).to(device)

    def forward(self,x):
        x1 = self.FFN(x)
        x2 = self.RELU(x1)
        x2 = x2.transpose(2, 1) #[bs,feature_dim,input_len]
        x3 = self.BN(x2)
        x3 = x3.transpose(2,1 ) #[bs,input_len,feature_dim]
        return x3


class FC_SL_BN(torch.nn.Module):
    def __init__(self,feature_dim,device):
        super().__init__()
        self.FFN = nn.Linear(feature_dim, feature_dim).to(device)
        self.RELU = nn.SiLU(inplace=False).to(device)
        self.BN = nn.BatchNorm1d(feature_dim,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True).to(device)

    def forward(self,x):
        x1 = self.FFN(x)
        x2 = self.RELU(x1)
        x2 = x2.transpose(2, 1) #[bs,feature_dim,input_len]
        x3 = self.BN(x2)
        x3 = x3.transpose(2, 1) #[bs,input_len,feature_dim]
        return x3

class FC_RE(torch.nn.Module):
    def __init__(self,feature_dim):
        super().__init__()
        self.FFN1 = nn.Linear(feature_dim, 128)
        self.RELU1 = nn.ReLU(inplace=False)
        self.FFN2 = nn.Linear(128, 64)
        self.RELU2 = nn.ReLU(inplace=False)
        self.FFN3 = nn.Linear(64, 32)
        self.RELU3 = nn.ReLU(inplace=False)

    def forward(self,x):
        x1 = self.FFN1(x)
        x2 = self.RELU1(x1)
        x3 = self.FFN2(x2)
        x4 = self.RELU2(x3)
        x5 = self.FFN3(x4)
        x6 = self.RELU3(x5)
        # x1 = self.FFN(x)
        # x2 = self.RELU(x1)
        return x6

class FC_SL(torch.nn.Module):
    def __init__(self,feature_dim):
        super().__init__()
        self.FFN1 = nn.Linear(feature_dim, 128)
        self.RELU1 = nn.SiLU(inplace=False)
        self.FFN2 = nn.Linear(128, 64)
        self.RELU2 = nn.SiLU(inplace=False)
        self.FFN3 = nn.Linear(64, 32)
        self.RELU3 = nn.SiLU(inplace=False)

    def forward(self,x):
        x1 = self.FFN1(x)
        x2 = self.RELU1(x1)
        x3 = self.FFN2(x2)
        x4 = self.RELU2(x3)
        x5 = self.FFN3(x4)
        x6 = self.RELU3(x5)
        # x1 = self.FFN(x)
        # x2 = self.RELU(x1)
        return x6