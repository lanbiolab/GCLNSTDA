import torch
import torch.nn as nn
from utils import sparse_dropout, spmm

import os
import h5py
import numpy as np

class GCLNSTDA(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, feature_path, feature_path_backups, prior, uninter_mat, device):
        super(GCLNSTDA,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.feature_path = feature_path
        self.feature_path_backups = feature_path_backups

        self.device = device

        self.num_negatives = 5
        self.alpha = 5
        self.prior = prior  # (|V|,)
        self.uninter_mat = uninter_mat
        self.num_items = 74


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    # negative sampling
    def bns(self, users,items, ui_scores):
        batch_size = users.size(0)
        if self.device == 'cpu':
            users = users.detach().numpy()
            ui_scores = ui_scores.detach().numpy()
        else:
            users = users.cpu().detach().numpy()
            ui_scores = ui_scores.cpu().detach().numpy()
        negatives = []
        for bs in range(batch_size):
            u = users[bs]
            i = items[bs]
            rating_vector = ui_scores[bs]
            x_ui = rating_vector[i]
            negative_items = self.uninter_mat[u]

            candidate_set = np.random.choice(negative_items, size=self.num_negatives, replace=False)
            candidate_scores = [rating_vector[l] for l in candidate_set]

            info = np.array([1 - self.sigmoid(x_ui - x_ul) for x_ul in candidate_scores])
            p_fn = np.array([self.prior[l] for l in candidate_set])
            F_n = np.array([np.sum(rating_vector <= x_ul) / (self.num_items+1) for x_ul in candidate_scores])
            unbias = (1 - F_n) * (1 - p_fn) / (1 - F_n - p_fn + 2 * F_n * p_fn)
            conditional_risk = (1 - unbias) * info - self.alpha * unbias * info
            j = candidate_set[conditional_risk.argsort()[0]]
            negatives.append(j)
        negatives = torch.LongTensor(negatives)
        negatives = negatives.to(self.device)
        return negatives

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:
            preds = self.E_u[uids] @ self.E_i.T

            user_feature = self.E_u[uids].detach().numpy()
            item_feature = self.E_i.detach().numpy()

            """
            添加：
            directory = os.path.dirname(self.feature_path)
            os.makedirs(directory, exist_ok=True)
            确保目录存在
            """

            directory = os.path.dirname(self.feature_path)
            os.makedirs(directory, exist_ok=True)

            with h5py.File(self.feature_path, 'w') as hf:
                hf['user_feature'] = user_feature
                hf['item_feature'] = item_feature

            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask)
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:
            if self.E_u != None:
                users = uids
                items = pos
                users_emb = self.E_u[users]
                items_emb = self.E_i
                ui_scores = torch.mm(users_emb, items_emb.t())
                negatives = self.bns(users, items, ui_scores)  # bs
                neg = negatives
                iids = torch.concat([pos, neg], dim=0)

            for layer in range(1,self.l+1):

                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))


                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)


                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)


            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()

            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            return loss, loss_r, self.lambda_1 * loss_s