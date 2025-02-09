from transformers import GPT2LMHeadModel, GPT2Tokenizer

import json
import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
import math
from os.path import join, abspath, dirname
import sys
import random
from model.modeling_t5_add_pos_eof import t5_for_rank_v2
from torch.utils.data import DataLoader, Dataset
from data.load_dataset import Feature_dataset,collate_fn
from mytransformers_emb_sim.src.transformers.models.t5.modeling_t5_mslr_add_eof import T5Config,T5ForRank_v2
from mytransformers_emb_sim.src.transformers import T5Tokenizer
SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large',
                  'megatron_11b']

k_max_src_len = 500
k_max_tgt_len = 5
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class T5DataSet(Dataset):
    def __init__(self, tokenizer, data, type_path, max_examples=-1,
                 max_src_len=200, max_tgt_len=200,batch_size=16):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len

        self.inputs = []            # list of dict
        self.targets = []           # list of dict
        self.input_text = []        # list of str
        self.target_text = []       # list of str
        self.batch_size = batch_size
        self._build()       # fill inputs, targets, max_lens

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        src_text = self.input_text[index]
        tgt_text = self.target_text[index]

        # These will be cast to torch.long in forward
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "source_text": src_text, "target_text": tgt_text}

    def _build(self):

        inputs_out = []     # accumulate the output of batch_encode
        targets_out = []    # same
        inputs_text = []    # save the original text for evaluations
        targets_text = []   # same


        for sample in self.data:
            # append end of sequence tokens (not necessary) because handled by tokenize() call
            src = sample['q1'] + '[SEP]' + sample['q2']
            if (sample['label'] == 'relevant'):
                tgt = 'Yes'
            else:
                tgt = 'No'
            # tgt = target[idx].strip()

            inputs_text.append(src)
            targets_text.append(tgt)

            # tokenize
            # padding="max_length" pads to max_len
            # otherwise (e.g. for batch), we could use padding=longest with truncation
            # note: don't need add_special_tokens since EOS added automatically and others are PAD
            # self.tokenizer returns a dict of input_ids and attention_masks (where attn masks corresponds to padding)
            # Note: padding could also be done via collate in dataloader
            # todo: we could actually batch encode these (i.e. multiple per)
            tokenized_inputs = self.tokenizer(
                [src], max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True
            )
            tokenized_targets = self.tokenizer(
                [tgt], max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
            )
            inputs_out.append(tokenized_inputs)
            targets_out.append(tokenized_targets)
        self.inputs = inputs_out
        self.targets = targets_out
        self.input_text = inputs_text
        self.target_text = targets_text

def get_dataloaders(tokenizer, batch_size, num_train, num_val, data, num_workers, shuffle_train=True,
                    shuffle_dev=False):
    """
    Returns: Tuple[train_loader : DataLoader, dev_loader : DataLoader]
    # Note:
    # - we default to not shuffling the dev set

    """
    # todo: should pass max src and max tgt len in as arguments
    train_data_set = T5DataSet(tokenizer, type_path="train", data=data, max_examples=num_train,
                               max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    eval_data_set = T5DataSet(tokenizer, type_path="val", data=data, max_examples=num_val,
                              max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    eval_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=shuffle_dev, num_workers=num_workers)

    return train_loader, eval_loader

class ShowProcess():

    i = 0
    max_steps = 0
    max_arrow = 50
    infoDone = 'done'

    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="P1001")
    parser.add_argument("--model_name", type=str, default='bert-base-cased')
    parser.add_argument("--doc_token", type=str, default='[DOC]')
    parser.add_argument("--doc_emb_token", type=str, default='[DOC_EMB]')
    parser.add_argument("--query_token", type=str, default='[QUERY]')
    parser.add_argument("--query_emb_token", type=str, default='[QUERY]')
    parser.add_argument("--start_token", type=str, default='[START]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--template", type=str, default="(3, 3, 5,0,0)")
    parser.add_argument("--early_stop", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=False)

    parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), './data/LAMA'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), './out/t5'))
    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str, default=join(abspath(dirname(__file__)), '../checkpoints'))

    args = parser.parse_args()

    # post-parsing args

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if self.args.model_name != 't5-11b' else 'cuda:{}'.format(self.args.t5_shard * 4)
        if self.args.use_original_template and (not self.args.use_lm_finetune) and (not self.args.only_evaluate):
            raise RuntimeError("""If use args.use_original_template is True, 
            either args.use_lm_finetune or args.only_evaluate should be True.""")

        import joblib
        self.train_data = joblib.load('./dataset/MSLR/FOLD1/processed_data/train_top40_without_process.pkl')
        self.dev_data = joblib.load('./dataset/MSLR/FOLD1/processed_data/test_top40_without_process.pkl')

        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_set = Feature_dataset(self.train_data)
        self.dev_set = Feature_dataset(self.dev_data)
        os.makedirs(self.get_save_path(), exist_ok=True)

        self.train_loader = DataLoader(self.train_set, batch_size=4, shuffle=True, drop_last=True,collate_fn=collate_fn)
        self.dev_loader = DataLoader(self.dev_set, batch_size=8,collate_fn=collate_fn)

        self.tokenizer = T5Tokenizer.from_pretrained('../T5-small', use_fast=False, local_files_only=True)
        self.config = T5Config.from_pretrained('../T5-small')
        self.feature_dim = 137
        self.config.num_layers = 2
        self.config.d_model = 256
        self.config.num_decoder_layers = 1
        self.t5_model = T5ForRank_v2(config=self.config, feature_dim=self.feature_dim)
        self.feature_table = joblib.load('./dataset/MSLR/FOLD1/processed_data/feature_top40_table_without_process.pkl')
        self.model = t5_for_rank_v2(self.args,self.device, self.feature_dim,self.feature_table,self.tokenizer,self.t5_model)
        self.out_path = './out_mslr_2_1/'

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            sum_ndcg_1 = 0
            sum_ndcg_5 = 0
            sum_ndcg_10 = 0
            sum_ndcg_jieduan = 0
            count = 0
            for batch in tqdm(self.dev_loader):
                count += len(batch[0])
                ndcg_1,ndcg_5,ndcg_10,ndcg_jieduan = self.model(batch[0], batch[1], x_ts=batch[2], label_doc_id=batch[3], label_input_id=None,id_label=batch[5], score=None, input_embeds_matrix=batch[5], evaluate=True)
                sum_ndcg_1 += ndcg_1
                sum_ndcg_5 += ndcg_5
                sum_ndcg_10 += ndcg_10
                sum_ndcg_jieduan += ndcg_jieduan
            print('NDCG@1 is {} NDCG@5 is {} NDCG@10 is {} DCGjieduan is {}'.format(sum_ndcg_1 / count,sum_ndcg_5 / count,sum_ndcg_10 / count,sum_ndcg_jieduan / count))
        return sum_ndcg_10 / count

    def get_task_name(self):
        if self.args.only_evaluate:
            return "_".join([self.args.model_name + ('_' + self.args.vocab_strategy), 'only_evaluate'])
        names = [self.args.model_name + ('_' + self.args.vocab_strategy),
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(),
                    self.args.relation_id)

    def get_checkpoint(self, epoch_idx, dev_f1s, dev_acc):
        ckpt_name = "epoch_{}_dev_f1{}_dev_acc{}.ckpt".format(epoch_idx, round(dev_f1s * 100, 4),
                                                              round(dev_acc * 100, 4))
        return {'embedding': self.model.state_dict(),
                'dev_f1s': dev_f1s,
                'dev_acc': dev_acc,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

    def train(self):
        best_ckpt = None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
        best_ndcg = 0
        low_epoch = 0
        for epoch_idx in range(100):
                tot_loss = 0
                print(len(self.train_loader))
                count = 0
                for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                    self.model.train()
                    optimizer.zero_grad()
                    if epoch_idx == 0:
                        for name, pram in self.model.named_parameters():
                            if 'layer.2' in name:
                                pram.requires_grad = False
                            else:
                                pram.requires_grad = True
                        loss = self.model(batch[0], batch[1],x_ts=batch[2],label_doc_id=batch[3],label_input_id=batch[4],id_label=batch[5],
                                          score=None,input_embeds_matrix=batch[5],evaluate=False,train_mode=0)
                    else:
                        if batch_idx % 2 == 0:
                            for name, pram in self.model.named_parameters():
                                if 'layer.2' in name:
                                   pram.requires_grad = False
                                else:
                                    pram.requires_grad = True
                            loss = self.model(batch[0], batch[1],x_ts=batch[2],label_doc_id=batch[3],label_input_id=batch[4],id_label=batch[5],
                                              score=None,input_embeds_matrix=batch[5],evaluate=False,train_mode=0)
                        else:
                            for name, pram in self.model.named_parameters():
                                if 'EncDecAttention' in name:
                                   pram.requires_grad = False
                                else:
                                    pram.requires_grad = True
                            loss = self.model(batch[0], batch[1],x_ts=batch[2],label_doc_id=batch[3],label_input_id=batch[4],id_label=batch[5],
                                              score=None,input_embeds_matrix=batch[5],evaluate=False,train_mode=1)
                    try:
                        print(loss.item())
                    except:
                        print('loss error')
                    tot_loss += loss.item()
                    count += 1
                    loss.backward()
                    optimizer.step()

                my_lr_scheduler.step()
                print('Epoch {} loss is {}'.format(epoch_idx,tot_loss / count))
                ndcg = self.evaluate()
                if(ndcg > best_ndcg):
                    print('epoch_{}_ndcg_{}.pth is saved'.format(epoch_idx,ndcg))
                    out_path = self.out_path + 'epoch_{}_ndcg_{}.pth'.format(epoch_idx,ndcg)
                    best_ndcg = ndcg
                else:
                    low_epoch += 1

        return best_ckpt

def main(relation_id=None):
    args = construct_generation_args()
    if relation_id:
        args.relation_id = relation_id
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()

