import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from SASRecModules_ori import *
import random
import json
import copy
from sklearn.metrics import roc_auc_score



logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='movie',
                        help='book, movie')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-5,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument("--model", type=str, default="SASRec",
                        help='the model name, GRU, Caser, SASRec')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument("--early_stop", type=int, default=20,
                        help='the epoch for early stop')
    parser.add_argument("--eval_num", type=int, default=1,
                        help='evaluate every eval_num epoch' )
    parser.add_argument("--seed", type=int, default=42,
                        help="the random seed")
    parser.add_argument("--randon_sample_num", type=int, default=64,
                        help="the random seed")
    parser.add_argument("--result_json_path", type=str, default="./result_temp/temp.json")
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        # indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        # state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output


class SASRec_with_label_CTR(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec_with_label_CTR, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        # embedding dim minors one for rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num,
            embedding_dim=hidden_size - 1,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()
    
    def get_mask(self, len_states, max_length):
        len_states = len_states.cuda()
        n_sequences = len_states.size(0)
        index_tensor = torch.arange(max_length).unsqueeze(0).repeat(n_sequences, 1).cuda()
        mask = (index_tensor < len_states.unsqueeze(1)).int().unsqueeze(-1)
        return mask


    def forward(self, states, state_rate, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        state_rate_reshape = state_rate.view(-1,20,1)
        inputs_emb = torch.cat((inputs_emb, state_rate_reshape), dim=2)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = self.get_mask(len_states, states.shape[1])
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, state_rate, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        state_rate_reshape = state_rate.view(-1,20,1)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = self.get_mask(len_states, states.shape[1])
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output



def evaluate(model, test_data, device):
    val_dataset = pd.read_csv(os.path.join(data_directory, test_data))
    val_dataset = RecDataset_seq_CTR(val_dataset, max_len)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8)

    with torch.no_grad():
        for j, (seq, history_rating, len_seq, target, target_rating) in enumerate(val_loader):
            model = model.eval()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            history_rating = history_rating.to(device)
            target_rating = target_rating.to(device)

            if args.model=="GRU":
                len_seq = len_seq.cpu()

            model_output = model.forward(seq, history_rating, len_seq)
            target = target.view((-1, 1))
            scores = (nn.Sigmoid()(torch.gather(model_output, 1, target))).view(-1).cpu()
            labels = target_rating.cpu()
            if j == 0:
                scores_all = scores
                labels_all = labels
            else:
                scores_all = torch.cat([scores_all, scores])
                labels_all = torch.cat([labels_all, labels])


    return roc_auc_score(labels_all, scores_all)


class RecDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df

    def __getitem__(self, i):
        temp = self.data.iloc[i]
        seq = torch.tensor(temp['seq'])
        len_seq = torch.tensor(temp['len_seq'])
        next = torch.tensor(temp['next'])
        return seq, len_seq, next

    def __len__(self):
        return len(self.data)

class RecDataset_seq_CTR(Dataset):
    def __init__(self, data_df, max_len):
        self.data = data_df
        self.max_len = max_len

    def __getitem__(self, i):
        temp = self.data.iloc[i]
        seq = torch.tensor(np.array(eval(temp['history_item_id']), dtype="int"))
        history_rating = torch.tensor(np.array(eval(temp['history_rating']), dtype="int"))
        # Padding only for taking place, use len_seq for mask while training and evaluating
        pad_len = self.max_len - len(seq)
        len_seq = len(seq)
        if pad_len != 0:
            pad_seq = seq[-1].repeat(pad_len)
            pad_rate = history_rating[-1].repeat(pad_len)
            seq = torch.cat([seq, pad_seq]) 
            history_rating = torch.cat([history_rating, pad_rate]) 

        next = torch.tensor(temp['item_id'])
        next_rating = torch.tensor(temp["rating"])
        
        return seq, history_rating, len_seq, next, next_rating

    def __len__(self):
        return len(self.data)


def main(result_folder, topk=[1,3,5,10,20]):
    if args.model=='SASRec':
        model = SASRec_with_label_CTR(args.hidden_factor,item_num, seq_size, args.dropout_rate, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer.to(device)

    train_data = pd.read_csv(os.path.join(data_directory, 'train.csv'))

    if args.randon_sample_num!=-1:
        train_data = train_data.sample(args.randon_sample_num)

    train_dataset = RecDataset_seq_CTR(train_data, max_len)
    print(f"Train dataset length: {len(train_dataset)}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)


    total_step=0
    auc_best = 0
    best_epoch = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size) + (int(num_rows/args.batch_size) * args.batch_size != num_rows)
    for i in range(args.epoch):
        # for j in tqdm(range(num_batches)):
        for j, (seq, history_rating, len_seq, target, target_rating) in tqdm(enumerate(train_loader)):
            model = model.train()
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            history_rating = history_rating.to(device)
            target_rating = target_rating.to(device)

            if args.model=="GRU":
                len_seq = len_seq.cpu()

            model_output = model.forward(seq, history_rating, len_seq)
            target = target.view((-1, 1))
            scores = nn.Sigmoid()(torch.gather(model_output, 1, target))

            # labels = torch.tensor(target_rating.to(device), dtype=float)
            labels = target_rating.to(device).float()

            loss = bce_loss(scores, labels.view(-1,1))

            loss_all = loss
            loss_all.backward()
            optimizer.step()

            if True:

                total_step+=1
                # if total_step % 200 == 0:
                #     print("the loss in %dth step is: %f" % (total_step, loss_all))
                #     # logging.info("the loss in %dth step is: %f" % (total_step, loss_all))

                if total_step % (num_batches * args.eval_num) == 0:

                        # print('VAL PHRASE:')
                        # logging.info('VAL PHRASE:')
                        eval_auc = evaluate(model, 'valid.csv', device)
                        print('TEST PHRASE:')
                        # logging.info('TEST PHRASE:')
                        test_auc = evaluate(model, 'test.csv', device)

                        model = model.train()

                        os.makedirs(result_folder, exist_ok=True)

                        if eval_auc > auc_best:

                            auc_best = eval_auc
                            best_epoch = i
                            early_stop = 0
                            test_result_at_best_eval = test_auc
                            best_model = copy.deepcopy(model)
                            torch.save(best_model, result_folder + "/best_model_" + str(args.l2_decay) + "_" +str(args.seed))


                        else:
                            early_stop += 1
                            if early_stop > args.early_stop:
                                return best_model, test_result_at_best_eval
                        
                        print('BEST EPOCH:{}'.format(best_epoch))
                        print('EARLY STOP:{}'.format(early_stop))
                        print("test_result_at_best_eval:")
                        print(test_result_at_best_eval)
    return best_model, test_result_at_best_eval
    


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.seed)

    data_directory = './data_for_CTR/' + args.data

    if args.data == "book":
        max_len = 10
        seq_size = max_len  # the length of history to define the seq
        item_num = 271380 # total number of items
    elif args.data == "movie":
        max_len = 20
        seq_size = max_len  # the length of history to define the seq
        item_num = 3706  # total number of items

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result_folder = ""
    for path_name in args.result_json_path.split("/")[:-1]:
        result_folder += path_name + "/"

    best_model, test_result_at_best_eval = main(result_folder)

