import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from gensim.models import Word2Vec
import json

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


def preprocess_data(data_dict, commit_list, special_token=False, out_file="preprocessed_data.pkl"):
    with open(data_path + data_dict['train'], 'r') as fp:
        ast_dict = json.load(fp)

    metrics = pd.read_csv(data_path + commit_list['train'])
    metrics['buggy'] = metrics['buggy'].apply(lambda x: 1 if x == True else 0)
    labels = metrics.set_index('commit_id')['buggy'].to_dict()
    c_list = metrics['commit_id'].tolist()

    metrics = metrics.drop(
        ['author_date', 'bugcount', 'fixcount', 'revd', 'tcmt', 'oexp',
         'orexp', 'osexp', 'osawr', 'project', 'buggy', 'fix'],
        axis=1, errors='ignore')
    metrics = metrics[['commit_id', 'la', 'ld', 'nf', 'nd', 'ns', 'ent',
                       'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp']]
    metrics = metrics.fillna(value=0)

    # 载入词向量模型
    files = list(data_dict['train']) + list(data_dict['val'])
    corpus = []
    for f_name in files:
        with open(data_path + f_name) as fp:
            subtrees = json.load(fp)
        for commit, files in subtrees.items():
            for f in files:
                for node_feature in f[1][0]:  # before subtree features
                    if not special_token:
                        corpus.append(node_feature)
                    else:
                        feature = node_feature[0]
                        if ':' in feature:
                            feature = node_feature[0].split(':')[0]
                        corpus.append(feature)
                for node_feature in f[2][0]:  # after subtree features
                    if not special_token:
                        corpus.append(node_feature)
                    else:
                        feature = node_feature[0]
                        if ':' in feature:
                            feature = node_feature[0].split(':')[0]
                        corpus.append(feature)

    corpus.append(['<UNK>'])
    vectorizer_model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

    preprocessed_data = []

    for c in c_list:
        try:
            commit = ast_dict.get(c, None)
        except Exception as e:
            print(f"Error loading commit data for {c}: {str(e)}")
            continue

        if commit is None:
            continue

        label = labels.get(c, 0)

        # 构建 metrics
        try:
            metrics_data = metrics[metrics['commit_id'] == c].drop(columns=['commit_id']).to_numpy(dtype=np.float32)[0,
                           :]
            metrics_tensor = torch.FloatTensor(metrics_data)
        except IndexError:
            metrics_tensor = torch.zeros(len(metrics.columns) - 1, dtype=torch.float)

        b_node_tokens, b_edges, b_colors = [], [[], []], []
        a_node_tokens, a_edges, a_colors = [], [[], []], []
        b_nodes_so_far, a_nodes_so_far = 0, 0

        for file in commit:
            b_node_tokens += [' '.join(node) for node in file[1][0]]
            b_colors += [c for c in file[1][2]]
            b_edges = [
                b_edges[0] + [s + b_nodes_so_far for s in file[1][1][0]],
                b_edges[1] + [d + b_nodes_so_far for d in file[1][1][1]]
            ]
            a_node_tokens += [' '.join(node) for node in file[2][0]]
            a_colors += [c for c in file[2][2]]
            a_edges = [
                a_edges[0] + [s + a_nodes_so_far for s in file[2][1][0]],
                a_edges[1] + [d + a_nodes_so_far for d in file[2][1][1]]
            ]

            b_n_nodes = len(file[1][0])
            a_n_nodes = len(file[2][0])
            b_nodes_so_far += b_n_nodes
            a_nodes_so_far += a_n_nodes

        before_embeddings = get_embedding(b_node_tokens, b_colors, vectorizer_model)
        before_adj = get_adjacency_matrix(b_nodes_so_far, b_edges[0], b_edges[1])
        after_embeddings = get_embedding(a_node_tokens, a_colors, vectorizer_model)
        after_adj = get_adjacency_matrix(a_nodes_so_far, a_edges[0], a_edges[1])

        training_data = [before_embeddings, before_adj, after_embeddings, after_adj, label, metrics_tensor]

        if b_nodes_so_far and a_nodes_so_far:
            preprocessed_data.append(training_data)

    # Save data to a file
    with open(data_path + out_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)


def get_adjacency_matrix(n_nodes, src, dst):
    """生成邻接矩阵"""
    edges = np.array([src, dst]).T
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for edge in edges:
        adj[edge[0], edge[1]] = 1
        adj[edge[1], edge[0]] = 1  # Ensure symmetry

    adj = np.vstack([adj, np.ones((1, adj.shape[1]), dtype=np.float32)])
    adj = np.hstack([adj, np.zeros((adj.shape[0], 1), dtype=np.float32)])
    adj = normalize(adj + np.eye(adj.shape[0]))
    return torch.FloatTensor(adj)


def get_embedding(file_node_tokens, colors, vectorizer_model):
    """获取词嵌入"""
    embeddings = []
    for i, node_feat in enumerate(file_node_tokens):
        if node_feat == 'None':
            colors.insert(i, 'lightgrey')
            assert colors[i] == 'lightgrey'
        if node_feat in vectorizer_model.wv:
            embeddings.append(vectorizer_model.wv[node_feat])
        else:
            embeddings.append(vectorizer_model.wv['<UNK>'])
    features = np.array(embeddings).astype(np.float32)

    # add color feature at the end of features
    color_feat = [1 if c == 'red' else
                  2 if c == 'green' else
                  3 if c == 'orange' else
                  4 if c == 'blue' else
                  0 for c in colors]

    features = np.hstack([features, np.array(color_feat, dtype=np.float32).reshape(-1, 1)])
    features = np.hstack([features, np.zeros((features.shape[0], 1), dtype=np.float32)])
    supernode_feat = np.zeros((1, features.shape[1]), dtype=np.float32)
    supernode_feat[-1, -1] = 1
    features = np.vstack([features, supernode_feat])

    features = normalize(features)
    return torch.FloatTensor(features)


def normalize(mx):
    """对矩阵进行归一化"""
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = np.dot(r_mat_inv, mx)
    return mx


class ASTDataset(Dataset):
    def __init__(self, preprocessed_file="preprocessed_data.pkl", data_path="data/"):
        self.data_path = data_path
        self.preprocessed_file = preprocessed_file
        self.preprocessed_data = self.load_preprocessed_data()

    def load_preprocessed_data(self):
        """从文件加载预处理的数据"""
        with open(self.data_path + self.preprocessed_file, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        """返回数据集的大小"""
        return len(self.preprocessed_data)

    def __getitem__(self, index):
        """返回单个样本"""
        return self.preprocessed_data[index]


# 使用示例
data_dict = {
    'train': ['train_file1.json', 'train_file2.json'],
    'val': ['val_file1.json', 'val_file2.json'],
}

commit_list = {
    'train': 'train_commits.csv',
    'val': 'val_commits.csv',
}

# 预处理并存储数据
preprocess_data(data_dict, commit_list, data_path="path_to_data/", preprocessed_file="preprocessed_data.pkl")

# 加载数据集
dataset = ASTDataset(preprocessed_file="preprocessed_data.pkl", data_path="path_to_data/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for batch in dataloader:
    before_embeddings, before_adj, after_embeddings, after_adj, label, metrics = batch
    # 在这里可以进行模型训练
