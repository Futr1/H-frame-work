import torch
from model.embedding_layer import Doc_embedding,Feature_embedding_pre_process_pos,FC_SL_BN
from transformers import T5Tokenizer
from torch.nn.utils.rnn import pad_sequence
import math
import torch.nn as nn
def get_dcg(label_list, k):
    dcgsum = 0
    for i in range(min(len(label_list), k)):
        dcg = (2 ** label_list[i] - 1) / math.log(i + 2, 2)
        dcgsum += dcg
    return dcgsum


def get_ndcg(pre_list, label_list,k):
    pre_score_list = []
    save_predict = []
    for i in pre_list:
        for j in label_list:
            if (i == j[0] and not i in save_predict):
                pre_score_list.append(j[1])
                save_predict.append(i)
    score_q = [j[1] for j in label_list]
    dcg = get_dcg(pre_score_list, k)
    idcg = get_dcg(score_q, k)
    ndcg = dcg / idcg
    return ndcg


def get_truncate_dcg(pre_list, label_list,k):
    pre_score_list = []
    for i in pre_list:
        for j in label_list:
            if (i == j[0]):
                pre_score_list.append(j[1])
    dcgsum = 0
    for i in range(min(len(pre_score_list), k)):
        if pre_score_list[i] == 0:
            score = -4
        elif pre_score_list[i] == 1:
            score = -2
        # elif pre_score_list[i] == 2:
        #     score = 0
        else:
            score = pre_score_list[i]
        dcg = score / math.log(i + 2, 2)
        dcgsum += dcg
    return dcgsum

class t5_for_rank_v2(torch.nn.Module):
    def __init__(self, args, device, feature_dim,feature_table,tokenizer,model):
        super().__init__()
        self.args = args
        self.device = device
        # load relation templates
        self.tokenizer = tokenizer
        self.model = model
        self.model = self.model.to(self.device)
        self.feature_table = feature_table


        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.doc_token,self.args.doc_emb_token,self.args.query_token,self.args.query_emb_token,self.args.start_token]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.doc_token_id = self.tokenizer.get_vocab()[self.args.doc_token]
        self.doc_emb_token_id = self.tokenizer.get_vocab()[self.args.doc_emb_token]
        self.query_token_id = self.tokenizer.get_vocab()[self.args.query_token]
        self.query_emb_token_id = self.tokenizer.get_vocab()[self.args.query_emb_token]
        self.start_emb_token_id = self.tokenizer.get_vocab()[self.args.start_token]
        self.embeddings = self.model.get_input_embeddings()
        self.hidden_size = self.embeddings.embedding_dim
        print(self.doc_emb_token_id,self.tokenizer.unk_token_id)

        self.pad_token_id = -100
        self.feature_dim = feature_dim
        self.transfer_layer = Feature_embedding_pre_process_pos(feature_dim=feature_dim,hidden_size=self.hidden_size,feature_table=feature_table,device=device)
        self.FRB_0 = FC_SL_BN(feature_dim=feature_dim,device=self.device)
        self.FRB_1 = FC_SL_BN(feature_dim=feature_dim, device=self.device)
        self.FRB_2 = FC_SL_BN(feature_dim=feature_dim, device=self.device)
        self.FC_out = nn.Linear(feature_dim,self.hidden_size).to(device)
        self.RELU_out = nn.ReLU(inplace=False).to(device)

    def embed_input(self, queries,doc_id_list,is_train):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.doc_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.doc_emb_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.query_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.query_emb_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.pad_token_id)] = self.tokenizer.unk_token_id

        raw_embeds = self.embeddings(queries_for_embedding)
        frb_embeds = raw_embeds.clone()
        frb_embeds = frb_embeds[:,:,:self.feature_dim]
        for bidx in range(bz):
            self.blocked_indices = (queries[bidx] == self.doc_emb_token_id).nonzero()[:,0]
            raw_embeds[bidx, self.blocked_indices, :],frb_embeds[bidx, self.blocked_indices, :] = self.transfer_layer(doc_id_list[bidx],is_train=is_train)
            # self.replace_embeds = self.transfer_layer(doc_id_list[i // 2])
            # raw_embeds[bidx, self.blocked_indices[bidx,i], :] = self.replace_embeds[i, :]
        frb_embeds = self.FRB_2(self.FRB_1(self.FRB_0(frb_embeds)))
        return raw_embeds,frb_embeds


    def embed_label_input(self, queries,doc_id_list):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.doc_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.doc_emb_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.query_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.query_emb_token_id)] = self.tokenizer.unk_token_id
        queries_for_embedding[(queries == self.pad_token_id)] = self.tokenizer.unk_token_id
        #print("embed_input:")
        #print(queries_for_embedding)
        raw_embeds = self.embeddings(queries_for_embedding)

        # For using handcraft prompts
        if self.args.use_original_template:
            return raw_embeds
        for bidx in range(bz):
            self.blocked_indices = (queries[bidx] == self.doc_emb_token_id).nonzero()[:,0]
            raw_embeds[bidx, self.blocked_indices, :] = self.transfer_layer(doc_id_list[bidx])

        return raw_embeds

    def get_query(self, x_h,is_label=False,is_predict=False):
        x_text = self.tokenizer.tokenize(x_h)
        if is_predict:
            return [[self.start_emb_token_id]]
        if not is_label:
            return [
                    self.tokenizer.convert_tokens_to_ids(x_text)
                    ]
        else:
            return [[self.start_emb_token_id]+
                    self.tokenizer.convert_tokens_to_ids(x_text)
                    ]


    def forward(self, x_hs, x_doc_id_list,x_ts,label_doc_id,label_input_id,id_label,input_embeds_matrix,score=None,evaluate=False,train_mode=None):
        bz = len(x_hs)
        if evaluate==False:

            queries = [torch.LongTensor(self.get_query(x_hs[i],is_label=False)).squeeze(0) for i in range(bz)]
            queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
            attention_mask = queries != self.pad_token_id
            inputs_embeds,frb_embeds = self.embed_input(queries,x_doc_id_list,is_train=True) #[bs,max_len_seq,hidden]


            tgt_ids = [torch.LongTensor(self.get_query(x_ts[i],is_label=True,is_predict=True)).squeeze(0) for i in range(bz)]
            tgt_ids = pad_sequence(tgt_ids, True, padding_value=self.pad_token_id).long().to(self.device)
            #tgt_ids[tgt_ids[:, :] == self.pad_token_id] = -10000
            label_ids = tgt_ids
            decoder_attention_mask = label_ids != self.pad_token_id
            # label_embeds = self.embed_label_input(label_ids,label_doc_id)
            label_embeds = self.embeddings(label_ids)

            labels = []
            for batch_item in x_doc_id_list:
                label_id_item = [label[3] for label in batch_item]
                labels.append(torch.tensor(label_id_item,dtype=torch.int))
            tgt_labels = pad_sequence(labels, True, padding_value=self.pad_token_id).long().to(self.device)
            tgt_labels[tgt_labels[:, :] == self.pad_token_id] = -10000

            labels_id = []
            for batch_item in label_input_id:
                label_id = [label for label in batch_item]
                labels_id.append(torch.tensor(label_id,dtype=torch.int))
            labels_id = pad_sequence(labels_id, True, padding_value=self.pad_token_id).long().to(self.device)
            labels_id[labels_id[:, :] == self.pad_token_id] = -10000

            loss = self.model(mode='train',inputs_embeds=inputs_embeds.to(self.device),attention_mask=attention_mask.to(self.device).bool(),
                                encoder_mask=attention_mask,labels=tgt_labels,labels_id=labels_id,decoder_inputs_embeds=label_embeds,
                                decoder_attention_mask=decoder_attention_mask, inputs_embeds_marix=inputs_embeds.to(self.device),
                                input_frb_embeds=frb_embeds,x_doc_id_list=x_doc_id_list,id_label=id_label,train_mode=train_mode,
                                decode_length=20,return_dict=True)
            return loss
        else:
            with torch.no_grad():
                queries = [torch.LongTensor(self.get_query(x_hs[i], is_label=False)).squeeze(0) for i in range(bz)]
                queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
                attention_mask = queries != self.pad_token_id
                inputs_embeds,frb_embeds = self.embed_input(queries,x_doc_id_list,is_train=False) #[bs,max_len_seq,hidden]


                tgt_ids = [torch.LongTensor(self.get_query(x_ts[i], is_label=True,is_predict=True)).squeeze(0)for i in range(bz)]
                tgt_ids = pad_sequence(tgt_ids, True, padding_value=self.pad_token_id).long().to(self.device)
                label_ids = tgt_ids
                decoder_attention_mask = label_ids != self.pad_token_id
                label_embeds = self.embeddings(label_ids)

                labels = []
                for batch_item in x_doc_id_list:
                    label_id_item = [label[3] for label in batch_item]
                    labels.append(torch.tensor(label_id_item, dtype=torch.int))
                tgt_labels = pad_sequence(labels, True, padding_value=self.pad_token_id).long().to(self.device)
                tgt_labels[tgt_labels[:, :] == self.pad_token_id] = -10000

                batch_pre,jieduan_idx = self.model(mode='generate',
                                     inputs_embeds=inputs_embeds.to(self.device),
                                     attention_mask=attention_mask.to(self.device).bool(),
                                     encoder_mask=attention_mask,input_frb_embeds=frb_embeds,
                                     labels=tgt_labels, decoder_inputs_embeds=label_embeds,
                                     decoder_attention_mask=decoder_attention_mask,decode_length=20,
                                     inputs_embeds_marix=inputs_embeds.to(self.device), return_dict=True)
                sum_ndcg_10 = 0
                sum_ndcg_5 = 0
                sum_ndcg_1 = 0
                sum_ndcg_jieduan = 0
                for i in range(bz):
                    label_seq = id_label[i]

                    pre_seq_idx = batch_pre[i]
                    pre_seq_idx_jieduan = batch_pre[i][:jieduan_idx[i]]
                    pre_seq = []
                    pre_seq_jieduan = []
                    pre = []
                    pre_jieduan = []
                    top_20 = [i[0] for i in label_seq]
                    top_20 = top_20[:35]
                    for idx in pre_seq_idx:
                        try:
                            pre_seq.append((x_doc_id_list[i][idx][1],x_doc_id_list[i][idx][3]))
                            pre.append(x_doc_id_list[i][idx][1])
                        except:
                            continue
                    for idx in pre_seq_idx_jieduan:
                        try:
                            pre_seq_jieduan.append((x_doc_id_list[i][idx][1],x_doc_id_list[i][idx][3]))
                            pre_jieduan.append(x_doc_id_list[i][idx][1])
                        except:
                            continue
                    if(label_seq[0][1] == 0):
                        continue
                    sum_ndcg_10 += get_ndcg(pre,label_seq,10)
                    sum_ndcg_5 += get_ndcg(pre,label_seq,5)
                    sum_ndcg_1 += get_ndcg(pre, label_seq, 1)
                    sum_ndcg_jieduan += get_truncate_dcg(pre_jieduan, label_seq, 40)
                return sum_ndcg_1,sum_ndcg_5,sum_ndcg_10,sum_ndcg_jieduan

