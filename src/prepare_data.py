import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from gensim.models import Word2Vec
import json
from tqdm import tqdm

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

data_lists = {
    'train': '/camel_train_1.json',
    'val': '/camel_val_1.json',
    'test': '/camel_test_1.json'
}
commit_lists = {
    'train': '/camel_train_filtered.csv',
    'val': '/camel_val_filtered.csv',
    'test': '/camel_test_filtered.csv'
}


def learn_word2vec(special_token=False):
    # 载入词向量模型
    files = list(data_lists['train']) + list(data_lists['val'])
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
    vectorizer_model = Word2Vec(corpus, vector_size=100, window=7, min_count=1, workers=4)

    print(len(vectorizer_model.wv.key_to_index))

    vectorizer_model.save('../trained_models/wv_camel.model')


wv_model = Word2Vec.load('../trained_models/wv_camel.model')


def preprocess_data(data_list, commit_list, out_file, mode='train'):
    with open(data_path + data_list[mode], 'r') as fp:
        ast_dict = json.load(fp)

    metrics = pd.read_csv(data_path + commit_list[mode])
    metrics['buggy'] = metrics['buggy'].apply(lambda x: 1 if x == True else 0)
    metrics['fix'] = metrics['fix'].apply(lambda x: 1 if x == True else 0)
    labels = metrics.set_index('commit_id')['buggy'].to_dict()
    c_list = metrics['commit_id'].tolist()

    # # clean for openstack and qt
    # metrics = metrics.drop(
    #     ['author_date', 'bugcount', 'fixcount', 'revd', 'nrev', 'rtime',
    #      'tcmt', 'hcmt', 'self', 'app', 'rexp', 'oexp', 'rrexp',
    #      'orexp', 'rsexp', 'osexp', 'asawr', 'osawr'],
    #     axis=1, errors='ignore')
    # metrics = metrics[['commit_id', 'la', 'ld', 'nf', 'nd', 'ns', 'ent',
    #                    'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp']]

    # clean for apachejit
    metrics = metrics.drop(
        ['project', 'buggy', 'year', 'author_date'], axis=1, errors='ignore')
    metrics = metrics[['commit_id', 'fix', 'la', 'ld', 'nf', 'nd', 'ns', 'ent',
                       'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp']]
    metrics = metrics.fillna(value=0)

    preprocessed_data = []

    for c in tqdm(c_list, desc='transform subtree', unit='commits'):
        try:
            commit = ast_dict.get(c, None)
        except Exception as e:
            print(f"Error loading commit data for {c}: {str(e)}")
            continue

        if commit is None:
            print('No commit data for {}'.format(c))
            continue

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

        if b_nodes_so_far + a_nodes_so_far > 29000 or b_nodes_so_far > 19000 or a_nodes_so_far > 19000:
            print()
            print('{} is a large commit, skip!'.format(c))
            continue

        before_embeddings = get_embedding(b_node_tokens, b_colors)
        before_adj = get_adjacency_matrix(b_nodes_so_far, b_edges[0], b_edges[1])
        after_embeddings = get_embedding(a_node_tokens, a_colors)
        after_adj = get_adjacency_matrix(a_nodes_so_far, a_edges[0], a_edges[1])

        label = labels[c]

        # 构建 metrics
        try:
            metrics_data = normalize(
                metrics[metrics['commit_id'] == c].drop(columns=['commit_id']).to_numpy(dtype=np.float32)[0, :])
            metrics_tensor = torch.FloatTensor(metrics_data)
        except IndexError:
            metrics_tensor = torch.zeros(len(metrics.columns) - 1, dtype=torch.float)

        training_data = [before_embeddings, before_adj, after_embeddings, after_adj, label, metrics_tensor]

        if b_nodes_so_far and a_nodes_so_far:
            preprocessed_data.append(training_data)

    # Save data to a file
    with h5py.File(data_path + out_file, 'w') as f:
        print('data len: {}'.format(len(preprocessed_data)))
        print('save data to {}'.format(data_path + out_file))
        # 创建数据集
        for idx, (before_embeddings, before_adj, after_embeddings, after_adj, label, metrics_tensor) in enumerate(
                preprocessed_data):
            # 假设每个元素是一个数组或列表，将它们转换为numpy数组
            f.create_dataset(f"before_embeddings_{idx}", data=np.array(before_embeddings))
            f.create_dataset(f"before_adj_{idx}", data=np.array(before_adj))
            f.create_dataset(f"after_embeddings_{idx}", data=np.array(after_embeddings))
            f.create_dataset(f"after_adj_{idx}", data=np.array(after_adj))
            f.create_dataset(f"label_{idx}", data=np.array(label))
            f.create_dataset(f"metrics_tensor_{idx}", data=np.array(metrics_tensor))


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


def get_embedding(file_node_tokens, colors):
    """获取词嵌入"""
    embeddings = []
    for i, node_feat in enumerate(file_node_tokens):
        if node_feat == 'None':
            colors.insert(i, 'lightgrey')
            assert colors[i] == 'lightgrey'
        if node_feat in wv_model.wv:
            embeddings.append(wv_model.wv[node_feat])
        else:
            embeddings.append(wv_model.wv['<UNK>'])
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
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.data_keys = list(self.file.keys())
        self.length = len(self.file[self.data_keys[0]])


    def __len__(self):
        """返回数据集的大小"""
        return self.length

    def __getitem__(self, index):
        # 获取每个数据项，按索引读取
        before_embeddings = np.array(self.file[f"before_embeddings_{index}"])
        before_adj = np.array(self.file[f"before_adj_{index}"])
        after_embeddings = np.array(self.file[f"after_embeddings_{index}"])
        after_adj = np.array(self.file[f"after_adj_{index}"])
        label = np.array(self.file[f"label_{index}"])
        metrics = np.array(self.file[f"metrics_tensor_{index}"])

        return before_embeddings, before_adj, after_embeddings, after_adj, label, metrics

    def close(self):
        self.file.close()


if __name__ == '__main__':
    print('--------start--------')
    # learn_word2vec()
    # word2vec = Word2Vec.load('../trained_models/wv_camel.model')
    # print(word2vec.wv['<UNK>'])
    # preprocess_data(data_lists, commit_lists, '/camel_train.h5', mode='val')
    val_dataset = ASTDataset(os.path.join(data_path, 'camel_train.h5'))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for data in val_loader:
        be, ba, ae, aa, l, m = data
        print(be.shape, ae.shape, aa.shape, l.shape, m.shape)
