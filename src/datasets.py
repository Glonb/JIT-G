import json
import os
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')
kaggle_path = '/kaggle/input/apache'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 768


class ASTDataset(Dataset):
    def __init__(self, data_dict, commit_list, special_token=True, transform=None):
        self.transform = transform
        self.special_token = special_token
        self.data_dict = data_dict
        self.commit_list = commit_list
        self.ast_dict = None
        self.c_list = None
        self.file_index = 0
        self.mode = 'train'
        self.vectorizer_model = None
        self.metrics = None
        self.labels = None
        self.load_metrics()
        # self.learn_vectorizer()
        self.learn_word2vec()

    def load_metrics(self):
        self.metrics = pd.read_csv(data_path + self.commit_list[self.mode])

        self.metrics['buggy'] = self.metrics['buggy'].apply(lambda x: 1 if x == True else 0)
        self.labels = self.metrics.set_index('commit_id')['buggy'].to_dict()

        self.c_list = self.metrics['commit_id'].tolist()

        self.metrics = self.metrics.drop(
            ['author_date', 'bugcount', 'fixcount', 'revd', 'tcmt', 'oexp',
             'orexp', 'osexp', 'osawr', 'project', 'buggy', 'fix'],
            axis=1, errors='ignore')
        self.metrics = self.metrics[['commit_id', 'la', 'ld', 'nf', 'nd', 'ns', 'ent',
                                     'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp']]
        self.metrics = self.metrics.fillna(value=0)

    # def learn_vectorizer(self):
    #     files = list(self.data_dict['train']) + list(self.data_dict['val'])
    #     corpus = []
    #     for f_name in files:
    #         with open(data_path + f_name) as fp:
    #             subtrees = json.load(fp)
    #         for commit, files in subtrees.items():
    #             for f in files:
    #                 for node_feature in f[1][0]:    # before subtree features
    #                     if len(node_feature) > 1:  # None
    #                         corpus.append(node_feature)
    #                     else:
    #                         if not self.special_token:
    #                             corpus.append(node_feature[0])
    #                         else:
    #                             feature = node_feature[0]
    #                             if ':' in node_feature[0]:
    #                                 feat_type = node_feature[0].split(':')[0]
    #                                 feature = feat_type + ' ' + '<' + feat_type[
    #                                                                   :3].upper() + '>'
    #                                 # e.g. number: 14 -> number <NUM>
    #                             corpus.append(feature)
    #                 for node_feature in f[2][0]:    # after subtree features
    #                     if len(node_feature) > 1:  # None
    #                         corpus.append(node_feature)
    #                     else:
    #                         if not self.special_token:
    #                             corpus.append(node_feature[0])
    #                         else:
    #                             feature = node_feature[0]
    #                             if ':' in node_feature[0]:
    #                                 feat_type = node_feature[0].split(':')[0]
    #                                 feature = feat_type + ' ' + '<' + feat_type[
    #                                                                   :3].upper() + '>'
    #                                 # e.g. number: 14 -> number <NUM>
    #                             corpus.append(feature)
    #
    #     vectorizer = CountVectorizer(lowercase=False, max_features=100000, binary=True)
    #     self.vectorizer_model = vectorizer.fit(corpus)
    #
    #     # with open(os.path.join(BASE_PATH, 'trained_models/vectorizer.pkl'), 'wb') as fp:
    #     #     pickle.dump(vectorizer, fp)

    def learn_word2vec(self):
        files = list(self.data_dict['train']) + list(self.data_dict['val'])
        corpus = []
        for f_name in files:
            with open(kaggle_path + f_name) as fp:
                subtrees = json.load(fp)
            for commit, files in subtrees.items():
                for f in files:
                    for node_feature in f[1][0]:  # before subtree features
                        if not self.special_token:
                            corpus.append(node_feature)
                        else:
                            feature = node_feature[0]
                            if ':' in feature:
                                feature = node_feature[0].split(':')[0]
                            corpus.append(feature)
                    for node_feature in f[2][0]:  # after subtree features
                        if not self.special_token:
                            corpus.append(node_feature)
                        else:
                            feature = node_feature[0]
                            if ':' in feature:
                                feature = node_feature[0].split(':')[0]
                            corpus.append(feature)

        corpus.append(['<UNK>'])
        model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
        self.vectorizer_model = model

        # with open(os.path.join(BASE_PATH, 'trained_models/vectorizer.pkl'), 'wb') as fp:
        #     pickle.dump(vectorizer, fp)

    def set_mode(self, mode):
        self.mode = mode
        # self.c_list = pd.read_csv(data_path + self.commit_list[self.mode])['commit_id'].tolist()
        self.load_metrics()
        self.file_index = 0
        with open(kaggle_path + self.data_dict[self.mode][self.file_index], 'r') as fp:
            self.ast_dict = json.load(fp)

    def switch_datafile(self):
        self.file_index += 1
        self.file_index %= len(self.data_dict[self.mode])
        with open(kaggle_path + self.data_dict[self.mode][self.file_index], 'r') as fp:
            self.ast_dict = json.load(fp)

    @staticmethod
    def normalize(mx):
        """Row-normalize sparse matrix"""
        row_sum = np.array(mx.sum(1))
        r_inv = np.power(row_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        # mx = r_mat_inv.dot(mx)
        mx = np.dot(r_mat_inv, mx)
        return mx

    # @staticmethod
    # def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    #     """Convert a scipy sparse matrix to a torch sparse tensor."""
    #     sparse_mx = sparse_mx.tocoo().astype(np.float32)
    #     indices = torch.from_numpy(
    #         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    #     values = torch.from_numpy(sparse_mx.data)
    #     shape = torch.Size(sparse_mx.shape)
    #     return torch.sparse.FloatTensor(indices, values, shape)

    def get_adjacency_matrix(self, n_nodes, src, dst):
        edges = np.array([src, dst]).T
        # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #                     shape=(n_nodes, n_nodes),
        #                     dtype=np.float32)

        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        # 填充邻接矩阵
        for edge in edges:
            adj[edge[0], edge[1]] = 1
            adj[edge[1], edge[0]] = 1  # 保证对称

        # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # add supernode
        adj = np.vstack([adj, np.ones((1, adj.shape[1]), dtype=np.float32)])
        adj = np.hstack([adj, np.zeros((adj.shape[0], 1), dtype=np.float32)])
        adj = self.normalize(adj + np.eye(adj.shape[0]))
        # adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        adj = torch.FloatTensor(adj)
        return adj

    def get_embedding(self, file_node_tokens, colors):
        # for i, node_feat in enumerate(file_node_tokens):
        #     file_node_tokens[i] = node_feat.strip()
        #     if node_feat == 'N o n e':
        #         file_node_tokens[i] = 'None'
        #         colors.insert(i, 'lightgrey')
        #         assert colors[i] == 'lightgrey'
        #     if self.special_token:
        #         if ':' in node_feat:
        #             feat_type = node_feat.split(':')[0]
        #             file_node_tokens[i] = feat_type + ' ' + '<' + feat_type[
        #                                                           :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
        # # fix the data later to remove the code above.
        # features = self.vectorizer_model.transform(file_node_tokens).astype(np.float32)

        embeddings = []
        for i, node_feat in enumerate(file_node_tokens):
            if node_feat == 'None':
                colors.insert(i, 'lightgrey')
                assert colors[i] == 'lightgrey'
            # 获取每个词的词向量，如果词不在词汇表中，使用特殊向量
            if node_feat in self.vectorizer_model.wv:
                embeddings.append(self.vectorizer_model.wv[node_feat])
            else:
                embeddings.append(self.vectorizer_model.wv['<UNK>'])  # 使用特殊向量填充
        features = np.array(embeddings).astype(np.float32)

        # print(features.shape)

        # add color feature at the end of features
        color_feat = [1 if c == 'red' else
                      2 if c == 'green' else
                      3 if c == 'orange' else
                      4 if c == 'blue' else
                      0 for c in colors]
        # print(np.array(color_feat).reshape(-1,1).shape)
        features = np.hstack([features, np.array(color_feat, dtype=np.float32).reshape(-1, 1)])
        # add supernode
        features = np.hstack([features, np.zeros((features.shape[0], 1), dtype=np.float32)])
        supernode_feat = np.zeros((1, features.shape[1]), dtype=np.float32)
        supernode_feat[-1, -1] = 1
        features = np.vstack([features, supernode_feat])

        features = self.normalize(features)
        # features = torch.FloatTensor(np.array(features.todense()))
        # return features
        return torch.FloatTensor(features)

    def __len__(self):
        return len(self.c_list)

    def __getitem__(self, item):
        c = self.c_list[item]
        while True:
            try:
                commit = self.ast_dict[c]
                break
            except:
                self.switch_datafile()
        label = self.labels[c]
        try:
            metrics = torch.FloatTensor(self.normalize(self.metrics[self.metrics['commit_id'] == c]
                                                       .drop(columns=['commit_id']).to_numpy(dtype=np.float32))[0, :])
        except IndexError:
            # commit id not in commit metric set
            dim = self.metrics[self.metrics['commit_id'] == c].drop(columns=['commit_id']).shape[1]
            metrics = torch.zeros(dim, dtype=torch.float)

        b_node_tokens, b_edges, b_colors = [], [[], []], []
        a_node_tokens, a_edges, a_colors = [], [[], []], []
        b_nodes_so_far, a_nodes_so_far = 0, 0
        for file in commit:
            b_node_tokens += [' '.join(node) for node in file[1][0]]
            b_colors += [c for c in file[1][2]]
            b_edges = [
                b_edges[0] + [s + b_nodes_so_far for s in file[1][1][0]],   # source nodes
                b_edges[1] + [d + b_nodes_so_far for d in file[1][1][1]]    # destination nodes
            ]
            a_node_tokens += [' '.join(node) for node in file[2][0]]
            a_colors += [c for c in file[2][2]]
            a_edges = [
                a_edges[0] + [s + a_nodes_so_far for s in file[2][1][0]],   # source nodes
                a_edges[1] + [d + a_nodes_so_far for d in file[2][1][1]]    # destination nodes
            ]

            b_n_nodes = len(file[1][0])
            a_n_nodes = len(file[2][0])
            b_nodes_so_far += b_n_nodes
            a_nodes_so_far += a_n_nodes

        if b_nodes_so_far + a_nodes_so_far > 29000 or b_nodes_so_far > 19000 or a_nodes_so_far > 19000:
            print('{} is a large commit, skip!'.format(c))
            return None
        # print(np.array(b_node_tokens).shape)
        # print(np.array(b_colors).shape)
        # print(b_node_tokens)
        before_embeddings = self.get_embedding(b_node_tokens, b_colors)
        before_adj = self.get_adjacency_matrix(b_nodes_so_far, b_edges[0], b_edges[1])
        after_embeddings = self.get_embedding(a_node_tokens, a_colors)
        after_adj = self.get_adjacency_matrix(a_nodes_so_far, a_edges[0], a_edges[1])
        training_data = [before_embeddings, before_adj, after_embeddings, after_adj, label, metrics]

        if not b_nodes_so_far and not a_nodes_so_far:
            print('commit {} has no file tensors.'.format(c))

        return training_data


if __name__ == "__main__":
    # data_dicts = {
    #     'train': ['/camel_train_1.json'],
    #     'val': ['/camel_val_1.json'],
    #     'test': ['/camel_val_1.json'],
    # }
    # commit_lists = {
    #     'train': '/camel_small.csv',
    #     'val': '/camel_val.csv',
    #     'test': '/camel_val.csv'
    # }
    # ast_dataset = ASTDataset(data_dict=data_dicts, commit_list=commit_lists, special_token=False)
    # ast_dataset.set_mode('train')
    # train_loader = DataLoader(ast_dataset, batch_size=1, shuffle=False)
    # print(ast_dataset.__len__())
    # print(ast_dataset.metrics.shape[1] - 1)
    # ast_dataset.set_mode('val')
    # print(ast_dataset.__len__())
    # train_iter = iter(train_loader)
    # for batch in train_iter:
    #     print(batch)
    print()
