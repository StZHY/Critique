import random
import os
import sys
import torch
import numpy as np

from time import time
from datetime import datetime
from prettytable import PrettyTable
from tqdm import tqdm

from generate_cri import generate_cri_key

from utils.parser import parse_args
from utils.cri_data_loader import load_data, load_kg
from modules.RISC import Critique
from utils.cri_evaluate import test


os.chdir(sys.path[0])

class Logger(object):
    def __init__(self, logFile = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def cal_pos_prob4replay(train_user_set, entity_neighbors):
    train_replay_prob = {}

    for user, train_items in train_user_set.items():

        all_items_neighbor = []
        for item in train_items:
            item_neili = list(entity_neighbors[item])
            all_items_neighbor.append(item_neili)
        
        all_neighbors_set = list(set([item for sublist in all_items_neighbor for item in sublist]))
        
        replay_jaccard_score = []
        for item_neighbor in all_items_neighbor:
            intersec = set(item_neighbor).intersection(all_neighbors_set)
            comb = set(all_neighbors_set + item_neighbor)
            jaccard_score = len(intersec) / len(comb)
            replay_jaccard_score.append(jaccard_score)

        replay_jaccard_score = np.array(replay_jaccard_score)
        replay_jaccard_score = replay_jaccard_score / replay_jaccard_score.sum()
        train_replay_prob[user] = replay_jaccard_score
    
    return train_replay_prob


def cosine_importance_sampling(cri_key, entity2item, entity_emb, user_emb):
    cri_user_cf = []

    for user, keys in cri_key.items():
        for key in keys:

            if key not in entity2item:
                continue

            around_items = entity2item[key]
            
            entity_tensor = entity_emb.weight[key].unsqueeze(0).detach().cpu()
            items_tensor = entity_emb.weight[around_items].detach().cpu()
            user_tensor = user_emb.weight[user].unsqueeze(0).detach().cpu()

            cosine_sim = torch.nn.functional.cosine_similarity(entity_tensor, items_tensor, dim=1)
            y_uv = torch.log(1 - torch.nn.Sigmoid()(torch.mul(user_tensor, items_tensor)))
            score = torch.abs(torch.mean(y_uv, dim =-1))
            p_1 = torch.mul(cosine_sim, score)
            p_1 = torch.clamp(p_1 / torch.sum(p_1), min=0)

            sampler = torch.distributions.Categorical(p_1)
            #samples = sampler(around_items)
            for i in range(args.rand_item_num):
                sample_item = around_items[sampler.sample()]
                cri_user_cf.append([user, sample_item])

    return cri_user_cf

def gat_importance_sampling(cri_key, entity2item, entity_emb, user_emb):
    cri_user_cf = []

    for user, keys in cri_key.items():
        for key in keys:

            if key not in entity2item:
                continue

            around_items = entity2item[key]

            entity_tensor = entity_emb.weight[key].unsqueeze(0).detach().cpu()
            items_tensor = entity_emb.weight[around_items].detach().cpu()
            user_tensor = user_emb.weight[user].unsqueeze(0).detach().cpu()


            attention = torch.nn.Sigmoid()(torch.mul(entity_tensor, items_tensor))
            attention_norm = torch.mean(torch.nn.functional.softmax(attention, dim=0), dim=-1)

            y_uv = torch.log(1 - torch.nn.Sigmoid()(torch.mul(user_tensor, items_tensor)))
            score = torch.abs(torch.mean(y_uv, dim =-1))

            q_1 = torch.mul(attention_norm, score)
            q_1 = q_1 / torch.sum(q_1)

            sampler = torch.distributions.Categorical(logits=q_1)

            for i in range(args.rand_item_num):
                sample_item = around_items[sampler.sample()]
                cri_user_cf.append([user, sample_item])

    return cri_user_cf 
    

def replay_get_pos_cf(neg_cf, n_params):

    pos_cf = []
    for user, neg in neg_cf:
        pos_item = int(user) + n_params['n_entities']
        pos_cf.append([user, pos_item, neg])
    return pos_cf

def feed_dict(cf_pairs, start, end):

    f_dict = {}

    entity_pairs = cf_pairs[start:end].to(device)
    f_dict['users'] = entity_pairs[:, 0]
    f_dict['pos'] = entity_pairs[:, 1]
    f_dict['neg'] = entity_pairs[:, 2]

    return f_dict



if __name__ == "__main__":

    """fix the random seed"""
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """log"""
    if args.training_log:
        now = datetime.now().strftime("%Y%m%d_%H%M")
        dataset_name = args.dataset
        sys.stdout = Logger('training_log/'+ dataset_name + ' ' + now + '.log')

    """print args important info"""
    print(os.path.basename(__file__))
    print("cri_lr: " + str(args.cri_lr) + " cri_key_rank_num: " + str(args.cri_key_rank_num) + " rand_item_num: " + str(args.rand_item_num) \
        + " imp_sample: "+str(args.imp_sample) + " replay_decay " + str(args.replay_decay))

    """load train_data & test data & user_dict & params"""
    train_cf, test_cf, user_dict = load_data(args)
    
    """load knowledge graph & key-item & item-key dict"""
    kg_graph, entity2item, item2entity, items_entity_mat, n_params, entity_neighbors = load_kg(args)
    
    """calculate pos items sample prob for replay and generate the [user, pos, neg] cf"""
    pos_prob4replay = cal_pos_prob4replay(user_dict['train_user_set'], entity_neighbors)
    
    """load base model"""
    kgin_save_path = 'weights/model_last-fm.pkl'
    kgin_model = torch.load(kgin_save_path, map_location=device)

    entity_embedding, user_embedding = kgin_model.generate()

    """critiquing model"""
    cri_model = Critique(args, user_embedding, entity_embedding, user_dict['train_user_set'], pos_prob4replay).to(device)
    user_emb = cri_model.user_emb
    entity_emb = cri_model.entity_emb

    # show the scores of based kgin model
    base_s_t = time()
    ret = test(cri_model, user_dict, n_params)
    base_e_t = time()

    base_res = PrettyTable()
    base_res.field_names = ["evaluating time", "recall", "ndcg", "precision", "hit_ratio"]
    base_res.add_row(
        [base_e_t - base_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
    )
    print(base_res)

    """cri model begin train"""
    cri_model.train()
    cri_optimizer = torch.optim.Adam(cri_model.parameters(), lr=args.cri_lr)
    
    for cri_epoch in range(args.cri_epoch):
        
        """ start critiquing """
        simulate_critiquing_s_t = time()

        """ simulate generating users' critique keyphrases """
        critiquing_key = generate_cri_key(cri_model, user_dict, n_params, items_entity_mat, args)

        """ important sampling for users' critiquing keys, here the items is neg items"""
        if args.imp_sample == 'cosine':
            important_sample_cf = cosine_importance_sampling(critiquing_key, entity2item, entity_emb, user_emb)
        if args.imp_sample == 'gat':
            important_sample_cf = gat_importance_sampling(critiquing_key, entity2item, entity_emb, user_emb)

        simulate_critiquing_e_t = time()

        cri_cf = replay_get_pos_cf(important_sample_cf, n_params)

        cri_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in cri_cf], np.int32))


        index = np.arange(len(cri_cf_pairs))
        np.random.shuffle(index)
        cri_cf_pairs = cri_cf_pairs[index]

        cri_loss, cri_s = 0, 0
        while cri_s + args.cri_batch_size <= len(cri_cf_pairs):
                    
            cri_batch = feed_dict(cri_cf_pairs, cri_s, cri_s + args.cri_batch_size)
            
            bpr_loss = cri_model(cri_batch)

            cri_optimizer.zero_grad()
            bpr_loss.backward()
            cri_optimizer.step()

            cri_loss += bpr_loss
            cri_s += args.cri_batch_size

        simulate_score_s_t = time()
        simulate_ret = test(cri_model, user_dict, n_params)
        simulate_score_e_t = time()

        simulate_res = PrettyTable()
        simulate_res.field_names = ["Cri_Epoch", "critiquing time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
        simulate_res.add_row(
            [cri_epoch, simulate_critiquing_e_t - simulate_critiquing_s_t, simulate_score_e_t - simulate_score_s_t, cri_loss.item(),\
            simulate_ret['recall'], simulate_ret['ndcg'], simulate_ret['precision'], simulate_ret['hit_ratio']]
            )
        print(simulate_res)

