# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from sklearn import manifold
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class DIGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DIGCN, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.cold_start = 0
        self.dataset = dataset
        # self.construction = 'weighted_max'
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']
        self.rev_weight = config['rev_weight']
        self.mask_weight = config['mask_weight']
        self.temp = config['temp']
        # self.info_weight = config['info_weight']

        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.mm_f_adj = None
        self.mm_c_adj = None

        self.mlp = nn.Linear(3 * dim_x, 3 * dim_x)

        self.m_attn = nn.MultiheadAttention(dim_x, 1)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            # self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            # self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        if self.v2_feat is not None:
            self.image2_embedding = nn.Embedding.from_pretrained(self.v2_feat, freeze=False)
            # self.image2_trs = nn.Linear(self.v2_feat.shape[1], self.feat_embed_dim)
        if self.t2_feat is not None:
            self.text2_embedding = nn.Embedding.from_pretrained(self.t2_feat, freeze=False)
            # self.text2_trs = nn.Linear(self.t2_feat.shape[1], self.feat_embed_dim)


        mm_f_adj_file = os.path.join(dataset_path, 'mm_f_adj_{}.pt'.format(self.knn_k))
        mm_c_adj_file = os.path.join(dataset_path, 'mm_c_adj_{}.pt'.format(self.knn_k))
        if os.path.exists(mm_f_adj_file) and os.path.exists(mm_c_adj_file):
            self.mm_f_adj = torch.load(mm_f_adj_file)
            self.mm_c_adj = torch.load(mm_c_adj_file)
        else:
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_f_adj, mm_f_adj_file)
            if self.v2_feat is not None:
                _, image2_adj = self.get_knn_adj_mat(self.image2_embedding.weight.detach())
                self.mm_adj = image2_adj
            if self.t2_feat is not None:
                _, text2_adj = self.get_knn_adj_mat(self.text2_embedding.weight.detach())
                self.mm2_adj = text2_adj
            if self.v2_feat is not None and self.t2_feat is not None:
                self.mm2_adj = self.mm_image_weight * image2_adj + (1.0 - self.mm_image_weight) * text2_adj
                del text2_adj
                del image2_adj
            torch.save(self.mm_c_adj, mm_c_adj_file)

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)
        self.mm_f_balanced = nn.Parameter(
            torch.tensor([1., 1., 1.])).to(self.device)
        self.mm_c_balance = nn.Parameter(
            torch.tensor([0.5])).to(self.device)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))

        self.f_gcn = IFGCN(self.dataset, batch_size, num_user, num_item, dim_x, num_layer=self.num_layer,
                               dim_latent=64, device=self.device, features_id=self.id_feat,
                              features_v=self.v_feat, features_t=self.t_feat, balanced_para=self.mm_f_balanced)
        self.c_gcn = IFGCN(self.dataset, batch_size, num_user, num_item, dim_x, num_layer=self.num_layer,
                              dim_latent=64, device=self.device, features_id=self.id_feat,
                              features_v=self.v2_feat, features_t=self.t2_feat, balanced_para=self.mm_c_balanced)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_guide = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)


    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # cross gcn
        self.rep_f, self.preference_f = self.f_gcn(self.edge_index, self.id_feat, self.v_feat, self.t_feat)
        self.rep_c, self.preference_c = self.c_gcn(self.edge_index, self.id_feat, self.v2_feat, self.t2_feat)

        self.user_rep_f = self.rep_f[:self.num_user]
        self.item_rep_f = self.rep_f[self.num_user:]

        self.user_rep_c = self.rep_c[:self.num_user]
        self.item_rep_c = self.rep_c[self.num_user:]

        # item-item graph
        h_t_mm_f = self.item_rep_f
        for i in range(self.n_layers):
            h_t_mm_f = torch.sparse.mm(self.mm_f_adj, h_t_mm_f)
        self.user_rep_f = self.user_rep_f
        self.item_rep_f = self.item_rep_f + h_t_mm_f

        h_t_mm_c = self.item_rep_c
        for i in range(self.n_layers):
            h_t_mm_c = torch.sparse.mm(self.mm_c_adj, h_t_mm_c)
        self.user_rep_c = self.user_rep_c
        self.item_rep_c = self.item_rep_c + h_t_mm_c

        # final result
        self.result_embed_f = torch.cat((self.user_rep_f, self.item_rep_f), dim=0)
        self.result_embed_c = torch.cat((self.user_rep_c, self.item_rep_c), dim=0)
        self.result_embed_final = self.result_embed_c * (1 - self.mm_c_balance[0]) + self.result_embed_f * self.mm_f_balance[0]
        user_tensor = self.result_embed_final[user_nodes]
        pos_item_tensor = self.result_embed_final[pos_item_nodes]
        neg_item_tensor = self.result_embed_final[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def InfoNCE(self, view1, view2, temp):
        '''Info NCE loss'''
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss = (self.preference[user] ** 2).mean() if self.preference is not None else 0.0

        reg_loss = self.reg_weight * reg_embedding_loss
        reg_loss += self.reg_weight * (self.balanced ** 2).mean()
        # align two perspectives
        rev_loss = self.InfoNCE(self.result_embed, self.result_embed_guide, self.temp)

        return loss_value + reg_loss + rev_loss * self.rev_weight

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax

                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax

            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


class IFGCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, num_layer,
                 dim_latent=None, device=None, features_id=None, features_v=None, features_t=None, balanced_para=None):
        super(IFGCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat_id = features_id.size(1)
        self.dim_feat_v = features_v.size(1)
        self.dim_feat_t = features_t.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device
        self.balance = balanced_para

        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, 3*self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
        self.MLP1_id = nn.Linear(self.dim_feat_id, 4 * self.dim_latent)
        self.MLP2_id = nn.Linear(4 * self.dim_latent, self.dim_latent)
        self.MLP1_v = nn.Linear(self.dim_feat_v, 4 * self.dim_latent)
        self.MLP2_v = nn.Linear(4 * self.dim_latent, self.dim_latent)
        self.MLP1_t = nn.Linear(self.dim_feat_t, 4 * self.dim_latent)
        self.MLP2_t = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, edge_index, features_id, features_v, features_t):
        id_features = self.MLP2_id(F.leaky_relu(self.MLP1_id(features_id)))
        v_features = self.MLP2_v(F.leaky_relu(self.MLP1_v(features_v)))
        t_features = self.MLP2_t(F.leaky_relu(self.MLP1_t(features_t)))

        # c_features = (self.balance[0] * id_features + self.balance[1] * v_features + self.balance[2] * t_features) / 3
        id_rep_1 = torch.cat((self.preference, id_features), dim=0).to(self.device)
        v_rep_1 = torch.cat((self.preference, v_features), dim=0).to(self.device)
        t_rep_1 = torch.cat((self.preference, t_features), dim=0).to(self.device)

        id_rep_1 = F.normalize(id_rep_1).to(self.device)
        v_rep_1 = F.normalize(v_rep_1).to(self.device)
        t_rep_1 = F.normalize(t_rep_1).to(self.device)

        id_rep_2 = self.conv_embed_id(id_rep_1, edge_index)
        v_rep_2 = self.conv_embed_v(v_rep_1, edge_index)
        t_rep_2 = self.conv_embed_t(t_rep_1, edge_index)

        weight1_v = 1 - F.cosine_similarity(id_rep_2, v_rep_2).mean()
        weight1_t = 1 - F.cosine_similarity(id_rep_2, t_rep_2).mean()
        tau1 = weight1_v + weight1_t + 1

        id_rep_2 = id_rep_2 + weight1_v * v_rep_2 + weight1_t * t_rep_2
        id_rep_2 = id_rep_2 / tau1

        id_rep_3 = self.conv_embed_id(id_rep_2, edge_index)
        v_rep_3 = self.conv_embed_v(v_rep_2, edge_index)
        t_rep_3 = self.conv_embed_t(t_rep_2, edge_index)

        weight2_v = 1 - F.cosine_similarity(id_rep_3, v_rep_3).mean()
        weight2_t = 1 - F.cosine_similarity(id_rep_3, t_rep_3).mean()
        tau2 = weight2_v + weight2_t + 1

        id_rep_3 = id_rep_3 + + weight2_v * v_rep_3 + weight2_t * t_rep_3
        id_rep_3 = id_rep_3 / tau2

        x_hat = id_rep_3 + id_rep_2 + id_rep_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, **kwargs):
        super(Base_gcn, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


