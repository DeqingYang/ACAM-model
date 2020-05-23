#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from torch.nn import functional as F
import torch
import numpy as np

class KAAM(torch.nn.Module):
    def __init__(self, args, ent_embed, attr_embed, item_attr_set, loader, use_user_attn=True,use_item_attn=True, use_pretrain=True):
        super(KAAM,self).__init__()
        self.Super_parameter_construct(args, loader, use_user_attn, use_item_attn, use_pretrain)
        self.variable_construct(args, ent_embed, item_attr_set, attr_embed)
        self.optimizer(args)
        for name, p in self.named_parameters():
            if 'set' not in name:
                p.requires_grad = True

    def Super_parameter_construct(self, args, loader, use_user_attn, use_item_attn, use_pretrain):
        self.d=int(args.d)
        self.loader = loader
        self.gamma = args.gamma
        self.dataset_name = args.dataset
        self.use_user_attn = use_user_attn
        self.use_item_attn = use_item_attn
        self.use_pretrain = use_pretrain


    def variable_construct(self, args, ent_embed, item_attr_set, attr_embed):
        self.relation = torch.nn.Embedding(7, self.d, max_norm=0.5, norm_type=2).cuda()
        self.w_r = torch.nn.Embedding(7, self.d, max_norm=0.5, norm_type=2).cuda()

        if self.use_pretrain:
            self.embed_ent = torch.nn.Embedding.from_pretrained(ent_embed,max_norm=0.5, norm_type=2)
            self.embed_attr_set = torch.nn.Embedding.from_pretrained(item_attr_set)
            self.embed_attr = torch.nn.Embedding.from_pretrained(attr_embed,max_norm=0.5, norm_type=2)
        else:
            self.embed_ent = torch.nn.Embedding(args.num_items, self.d,max_norm=0.5, norm_type=2).cuda()
            self.embed_attr_set = torch.nn.Embedding.from_pretrained(item_attr_set)
            self.embed_attr = torch.nn.Embedding(args.num_attrs, self.d,max_norm=0.5, norm_type=2,).cuda()

        self.W1 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.d, self.d))).cuda()
        self.W2 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.d, self.d))).cuda()
        self.b1 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(5, self.d))).cuda()
        self.b2 = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(5, self.d))).cuda()

        self.predict = torch.nn.Sequential(torch.nn.Linear(self.d * 4, 96), torch.nn.BatchNorm1d(96),
                                           torch.nn.ReLU(),torch.nn.Linear(96, 20),
                                           torch.nn.ReLU(),torch.nn.Linear(20, 2)).cuda()
        self.attn_matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(5, 5))).cuda()

    def forward(self, seq):
        ent = self.embed_ent(seq).cuda()
        attr_set = self.embed_attr_set(seq).long().cuda()
        attr = self.embed_attr(attr_set).cuda()
        ent_hist, ent_cand = ent[:, :-1, :].cuda(), ent[:, -1, :].cuda()
        attr_hist, attr_cand = attr[:, :-1, :, :].cuda(), attr[:, -1, :, :].cuda()

        if self.dataset_name=='music' : attr_singer_hist, attr_singer_cand = attr_hist[:, :, 0, :].cuda(), attr_cand[:, 0, :].cuda()
        if self.dataset_name == 'movie': attr_singer_hist, attr_singer_cand = attr_hist[:, :, :4, :].mean(dim=2).cuda(), attr_cand[:, :4, :].mean(dim=1).cuda()

        attr_album_hist, attr_album_cand = attr_hist[:, :, -3, :].cuda(), attr_cand[:, -3, :].cuda()
        attr_composer_hist, attr_composer_cand = attr_hist[:, :, -2, :].cuda(), attr_cand[:, -2, :].cuda()
        attr_author_hist, attr_author_cand = attr_hist[:, :, -1, :].cuda(), attr_cand[:, -1, :].cuda()

        user = torch.stack((
        torch.mean(torch.tanh(ent_hist.bmm(ent_cand.repeat(1,1,1).permute(1,2,0)).permute(0,2,1)).bmm(ent_hist),1),
        torch.mean(torch.tanh(attr_singer_hist.bmm(attr_singer_cand.repeat(1, 1, 1).permute(1, 2, 0)).permute(0, 2, 1)).bmm(attr_singer_hist),1),
        torch.mean(torch.tanh(attr_album_hist.bmm(attr_album_cand.repeat(1,1,1).permute(1,2,0)).permute(0,2,1)).bmm(attr_album_hist),1),
        torch.mean(torch.tanh(attr_composer_hist.bmm(attr_composer_cand.repeat(1, 1, 1).permute(1, 2, 0)).permute(0, 2, 1)).bmm(attr_composer_hist), 1),
        torch.mean(torch.tanh(attr_author_hist.bmm(attr_author_cand.repeat(1, 1, 1).permute(1, 2, 0)).permute(0, 2, 1)).bmm(attr_author_hist), 1)), dim=1)

        item = torch.stack((ent_cand, attr_singer_cand, attr_album_cand, attr_composer_cand, attr_author_cand), dim=1)

        size = seq.shape[0]
        item_key =torch.tanh(item.bmm(self.W2.unsqueeze(0).repeat((size,1,1)))+self.b2.unsqueeze(0).repeat((size,1,1)))
        user_key =torch.tanh(user.bmm(self.W1.unsqueeze(0).repeat((size,1,1)))+self.b1.unsqueeze(0).repeat((size,1,1)))

        attn_map = user_key.bmm(item_key.permute((0, 2, 1)))
        self.attn_matrix = torch.sum(attn_map, 0)

        user_set = F.softmax(attn_map, dim=1).permute((0, 2, 1)).bmm(user_key) if self.use_user_attn else user_val
        item_set = F.softmax(attn_map, dim=2).bmm(item_key) if self.use_item_attn else item_val

        if self.use_user_attn and self.use_item_attn:
            y = torch.cat((torch.sum(user_set, 1), torch.sum(item_set, 1),torch.sum(user_key, 1),torch.sum(item_key, 1)), dim=1)

        else:
            y = torch.cat(
                (torch.sum(user_set, 1), torch.sum(item_set, 1), torch.sum(user_key, 1), torch.sum(item_key, 1)), dim=1)
        y = self.predict(y)

        kg_loss = 1

        if self.dataset_name=='music' :
            relation_type = torch.from_numpy(np.array([0,1,2,3])).cuda()
            ent_head = ent.repeat((4, 1, 1, 1)).permute(1, 2, 0, 3)
        if self.dataset_name == 'movie':
            relation_type = torch.from_numpy(np.array([0,0,0,0,1,2,3])).cuda()
            ent_head = ent.repeat((7, 1, 1, 1)).permute(1, 2, 0, 3)

        relation_emb = self.relation(relation_type)
        w_r_emb = self.w_r(relation_type)

        relation_matrix = relation_emb.repeat((attr.size()[0], attr.size()[1], 1, 1))
        w_r_matrix =  w_r_emb.repeat((attr.size()[0], attr.size()[1], 1, 1))

        wrong_attr = attr[torch.randperm(relation_matrix.size(0)),:,:,:]
        wrong_attr = wrong_attr[:,torch.randperm(relation_matrix.size(1)),:,:]
        wrong_attr = wrong_attr[:,:,torch.randperm(relation_matrix.size(2)),:]

        ent_head_p = w_r_matrix.mul(torch.sum(w_r_matrix.mul(ent_head),dim=-1).unsqueeze(-1).repeat((1,1,1,self.d)))
        attr_p = w_r_matrix.mul(torch.sum(w_r_matrix.mul(attr),dim=-1).unsqueeze(-1).repeat((1,1,1,self.d)))
        wrong_attr_p = w_r_matrix.mul(torch.sum(w_r_matrix.mul(wrong_attr),dim=-1).unsqueeze(-1).repeat((1,1,1,self.d)))

        d_right = ent_head_p + relation_matrix - attr_p
        d_wrong = ent_head_p + relation_matrix - wrong_attr_p
        score_tuple = torch.mean(d_right ** 2, dim=-1) ** 0.5 - torch.mean(d_wrong ** 2, dim=-1) ** 0.5
        kg_loss = torch.mean(score_tuple)
        return y,kg_loss

    def optimizer(self,args):
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.learning_rate, l2_reg = args.learning_rate, args.l2_reg
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=l2_reg)

    def calculate(self,args, train_set_length):
        E=np.zeros([5,5])
        for x, y in self.loader:
            y_pred, kg_loss = self.forward(x)
            E += self.attn_matrix.cpu().detach().numpy()
            loss = self.loss_func(y_pred, y) if not args.use_KGloss else self.loss_func(y_pred, y) + kg_loss
            # print('1:',self.loss_func(y_pred, y).cpu().detach().numpy())
            # print('2:',loss_kg.cpu().detach().numpy())
            self.train_op.zero_grad()
            loss.backward()
            self.train_op.step()
            np.save('../result/' + str(args.dataset) + str(args.L) + str(args.method) + 'result.npy', E / train_set_length)

    def learning_rate_adjust(self, args):
        self.learning_rate *= 0.8
        self.train_op = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=args.l2_reg)