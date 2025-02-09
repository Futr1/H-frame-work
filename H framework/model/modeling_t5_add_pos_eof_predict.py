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
        # for param in self.model.parameters():
        #     param.requires_grad = self.args.use_lm_finetune
        # for name,param in self.model.named_parameters():
        #     if 'word_embeddings' in name:
        #         param.requires_grad = False

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
        #print('pseudo_token_id')
        #print(self.pseudo_token_id)
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
        #print("embed_input:")
        #print(queries_for_embedding)
        raw_embeds = self.embeddings(queries_for_embedding)
        frb_embeds = raw_embeds.clone()
        frb_embeds = frb_embeds[:,:,:self.feature_dim]
        #print(raw_embeds)
        #print(blocked_indices)
        #self.blocked_indices = (queries == self.doc_emb_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :,1]  # bz
        # print(queries[:,16])
        # print(self.doc_emb_token_id) ##32101
        # print(self.doc_token_id) #32100
        # self.blocked_indices = (queries == self.doc_emb_token_id).nonzero().reshape((bz, self.spell_length, 2))
        # print(self.blocked_indices.shape)
        #self.replace_embeds = self.transfer_layer(doc_id_list)
        # replace_embeds = torch.cat([self.prompt_encoder_q1(),self.prompt_encoder_q2,self.prompt_encoder_prompt],dim=0)
        # print(replace_embeds)
        # 把输入中prompt token位置的向量替换成lstm+mlp的输出向量
        for bidx in range(bz):
            self.blocked_indices = (queries[bidx] == self.doc_emb_token_id).nonzero()[:,0]
            # print(self.blocked_indices)
            # print('len of input_feature_id in Feature_embedding is:')
            # print(len(doc_id_list[bidx]))
            # for i in self.blocked_indices:
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
            # print(self.blocked_indices)
            # print('len of input_feature_id in Feature_embedding is:')
            # print(len(doc_id_list[bidx]))
            # for i in self.blocked_indices:
            raw_embeds[bidx, self.blocked_indices, :] = self.transfer_layer(doc_id_list[bidx])
            # self.replace_embeds = self.transfer_layer(doc_id_list[i // 2])
            # raw_embeds[bidx, self.blocked_indices[bidx,i], :] = self.replace_embeds[i, :]

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
    # def get_fc_re_bn_out(self,raw_embeds):
    #     out1 = self.FC_RE_BN_0(raw_embeds)
    #     out2 = self.FC_RE_BN_1(out1)
    #     out3 = self.FC_RE_BN_2(out2)
    #     return out3

    def forward(self, x_hs, x_doc_id_list,x_ts,label_doc_id,label_input_id,id_label,input_embeds_matrix,score=None,evaluate=False,train_mode=None):
        bz = len(x_hs)
        if evaluate==False:
            # 将输入的[DOC_EMB]序列转化为对应的embeddings
            queries = [torch.LongTensor(self.get_query(x_hs[i],is_label=False)).squeeze(0) for i in range(bz)]
            queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
            attention_mask = queries != self.pad_token_id
            inputs_embeds,frb_embeds = self.embed_input(queries,x_doc_id_list,is_train=True) #[bs,max_len_seq,hidden]

            # 将输入的label [DOC_EMB]序列转化为相应的embeddings
            tgt_ids = [torch.LongTensor(self.get_query(x_ts[i],is_label=True,is_predict=True)).squeeze(0) for i in range(bz)]
            tgt_ids = pad_sequence(tgt_ids, True, padding_value=self.pad_token_id).long().to(self.device)
            #tgt_ids[tgt_ids[:, :] == self.pad_token_id] = -10000
            label_ids = tgt_ids
            decoder_attention_mask = label_ids != self.pad_token_id
            # label_embeds = self.embed_label_input(label_ids,label_doc_id)
            label_embeds = self.embeddings(label_ids)
            #获取输入batch中每条数据对应的label分数，后续计算其softmax
            labels = []
            for batch_item in x_doc_id_list:
                label_id_item = [label[3] for label in batch_item]
                labels.append(torch.tensor(label_id_item,dtype=torch.int))
            tgt_labels = pad_sequence(labels, True, padding_value=self.pad_token_id).long().to(self.device)
            tgt_labels[tgt_labels[:, :] == self.pad_token_id] = -10000
            ##获取输入batch中正确排序的label id，用于每一步计算loss时将前面已选择的id掩盖
            labels_id = []
            for batch_item in label_input_id:
                label_id = [label for label in batch_item]
                labels_id.append(torch.tensor(label_id,dtype=torch.int))
            labels_id = pad_sequence(labels_id, True, padding_value=self.pad_token_id).long().to(self.device)
            labels_id[labels_id[:, :] == self.pad_token_id] = -10000
            # print('label:')
            # print(tgt_ids)
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

                # 将输入的label [DOC_EMB]序列转化为相应的embeddings
                tgt_ids = [torch.LongTensor(self.get_query(x_ts[i], is_label=True,is_predict=True)).squeeze(0)for i in range(bz)]
                tgt_ids = pad_sequence(tgt_ids, True, padding_value=self.pad_token_id).long().to(self.device)
                label_ids = tgt_ids
                decoder_attention_mask = label_ids != self.pad_token_id
                label_embeds = self.embeddings(label_ids)
                # 获取输入batch中每条数据对应的label分数，后续计算其softmax
                labels = []
                for batch_item in x_doc_id_list:
                    label_id_item = [label[3] for label in batch_item]
                    labels.append(torch.tensor(label_id_item, dtype=torch.int))
                tgt_labels = pad_sequence(labels, True, padding_value=self.pad_token_id).long().to(self.device)
                tgt_labels[tgt_labels[:, :] == self.pad_token_id] = -10000
                # print('label:')
                # print(tgt_ids)
                batch_pre,step_rank_idx= self.model(mode='generate',
                                     inputs_embeds=inputs_embeds.to(self.device),
                                     attention_mask=attention_mask.to(self.device).bool(),
                                     encoder_mask=attention_mask,input_frb_embeds=frb_embeds,
                                     labels=tgt_labels, decoder_inputs_embeds=label_embeds,
                                     decoder_attention_mask=decoder_attention_mask,decode_length=20,
                                     inputs_embeds_marix=inputs_embeds.to(self.device), return_dict=True)
                sum_ndcg_10 = 0
                sum_ndcg_5 = 0
                sum_ndcg_1 = 0
                sum_ndcg_step_10 = [0]*30
                pre_seq_list = []
                for i in range(bz):
                    label_seq = id_label[i]
                    #这里好像有问题，id_label是doc_id，而输出序列是文档在列表中的索引，不能直接对应
                    ## 应该用x_doc_id_list，直接利用预测的索引对应label
                    ## idcg的话就用id_label，因为已经按照label排好序了
                    pre_seq_idx = batch_pre[i]
                    step_pre_rank_idx = step_rank_idx[i]
                    pre_seq = []
                    pre = []
                    top_20 = [i[0] for i in label_seq]
                    top_20 = top_20[:10]
                    for idx in pre_seq_idx:
                        try:
                            pre_seq.append((x_doc_id_list[i][idx][1],x_doc_id_list[i][idx][3]))
                            pre.append(x_doc_id_list[i][idx][1])
                        except:
                            continue
                    ## 计算每个step的ndcg
                    step_count = 0
                    for step in step_pre_rank_idx:
                        temp_pre = []
                        for idx in step:
                            try:
                                temp_pre.append(x_doc_id_list[i][idx][1])
                            except:
                                continue
                        # print(label_seq)
                        # print(temp_pre)
                        if (label_seq[0][1] == 0):
                            continue
                        sum_ndcg_step_10[step_count] += get_ndcg(temp_pre,label_seq,10)
                        step_count += 1
                    #################
                    # print(top_20)
                    # print(pre_seq)
                    #pre_seq_list.append(pre_seq)

                    if(label_seq[0][1] == 0):
                        continue
                    sum_ndcg_10 += get_ndcg(pre,label_seq,10)
                    sum_ndcg_5 += get_ndcg(pre,label_seq,5)
                    sum_ndcg_1 += get_ndcg(pre, label_seq, 1)
                #joblib.dump(pre_seq_list,'/data/xushicheng/seq2rank/out/pre_seq_istella1024')
                return sum_ndcg_1,sum_ndcg_5,sum_ndcg_10,sum_ndcg_step_10


                # 将输入的label [DOC_EMB]序列转化为相应的embeddings
                # tgt_ids = [torch.LongTensor(self.get_query(x_ts[i], is_label=True)).squeeze(0) for i in range(bz)]
                # tgt_ids = pad_sequence(tgt_ids, True, padding_value=self.pad_token_id).long().to(self.device)
                # tgt_ids[tgt_ids[:, :] == self.pad_token_id] = -10000
                # label_ids = tgt_ids
                # label_embeds = self.embed_label_input(label_ids, label_doc_id)
                # # 获取输入batch中每条数据对应的label分数，后续计算其softmax
                # labels = []
                # for batch_item in x_doc_id_list:
                #     label_id_item = [label[3] for label in batch_item]
                #     labels.append(torch.tensor(label_id_item, dtype=torch.int))
                # tgt_labels = pad_sequence(labels, True, padding_value=self.pad_token_id).long().to(self.device)
                # tgt_labels[tgt_labels[:, :] == self.pad_token_id] = -10000

                # out_ids = self.model.generate(inputs_embeds=inputs_embeds.to(self.device),attention_mask=attention_mask.to(self.device))
                # out_seq_batch = self.tokenizer.batch_decode(out_ids, skip_special_tokens=False)
                # label_seqs = x_ts
                # sum_ndcg = 0
                # for i in range(bz):
                #     score_q = score[i]
                #     label_seq = label_seqs[i].split('[DOC]')[1:]
                #     pre_seq = out_seq_batch[i].split('[DOC]')[1:]
                #     print(label_seq)
                #     print(score_q)
                #     print(pre_seq)
                #     if(score_q[0] == 0):
                #         continue
                #     sum_ndcg += get_ndcg(pre_seq,label_seq,score_q,10)
                # return sum_ndcg / bz
        #
        # for i in range(bz):
        #     answer_idx = 0
        #     # print(pred_ids[i][answer_idx][0].item())
        #     pred_seq = pred_ids[i, answer_idx].tolist()
        #     # print(pred_seq[0])
        #     top_50 = []
        #     for pred in pred_seq:
        #         # if pred in self.allowed_vocab_ids:
        #         top_50.append(pred)
        #         if (len(top_50) >= 10):
        #             break
