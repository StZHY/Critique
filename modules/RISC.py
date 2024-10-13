import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Critique(nn.Module):
    def __init__(self, args_config, user_embedding, entity_embedding, train_user_set, pos_prob4replay):
        super(Critique, self).__init__()

        self.device = torch.device("cuda:"+str(args_config.gpu_id)) if args_config.cuda else torch.device("cpu")

        self.emb_size = args_config.dim

        self.user_emb = nn.Embedding.from_pretrained(user_embedding, freeze=False)
        self.entity_emb = entity_embedding

        self.supplement_average_items(train_user_set, pos_prob4replay)

    def supplement_average_items(self, train_user_set, pos_prob4replay):

        user_num = len(train_user_set)
        aver_items_tensor_all = torch.empty(user_num, self.emb_size).to(self.device)

        for user, items in train_user_set.items():
            items_weight = torch.tensor(pos_prob4replay[user]).unsqueeze(1).to(self.device)
            
            items = torch.tensor(items, dtype=int)
            items_tensor = self.entity_emb[items]
            aver_items_tensor = torch.sum((items_weight* items_tensor), dim=0)
            aver_items_tensor_all[user] = aver_items_tensor

        self.entity_emb = torch.cat((self.entity_emb, aver_items_tensor_all), dim=0)
        self.entity_emb = nn.Embedding.from_pretrained(self.entity_emb, freeze=True)

    def forward(self, batch):
        user_id = batch['users']
        item_id = batch['pos']
        keyphrase_id = batch['neg']

        user_emb_input = self.user_emb(user_id)
        item_emb_input = self.entity_emb(item_id)
        keyphrase_emb_input = self.entity_emb(keyphrase_id)

        bpr_loss = self.create_bpr_loss(user_emb_input, item_emb_input, keyphrase_emb_input)

        return bpr_loss
    
    def create_bpr_loss(self, user_emb, item_emb, key_emb):
        pos_scores = user_emb * item_emb
        neg_scores = user_emb * key_emb

        bpr_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores) + torch.log(1 - nn.Sigmoid()(neg_scores)) )

        return bpr_loss
    
    def create_output(self, batch):
        user_id = batch['users']
        item_id = batch['items']

        user_emb_input = self.user_emb(user_id)
        entity_emb_input = self.entity_emb(item_id)

        score = torch.mul(user_emb_input, entity_emb_input).sum(dim=1)

        return score
    
    def generate(self):
        return self.user_emb, self.entity_emb
    
    def rating(self, u_embeddings, i_embeddings):
        return torch.matmul(u_embeddings, i_embeddings.t())