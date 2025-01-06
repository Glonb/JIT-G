# from sklearn.feature_extraction.text import CountVectorizer
#
# corpus = [
#     'public SwitchStatement(Expression expression, Statement defaultStatement) {'
#     'this.expression = expression;',
#     'this.defaultStatement = defaultStatement;',
#     '}',
#     'public void visit(GroovyCodeVisitor visitor) {',
#     'visitor.visitSwitch(this);',
#     '}',
# ]
#
# vectorizer = CountVectorizer(lowercase=False, binary=True)
#
# x = vectorizer.fit_transform(corpus)
#
# print(vectorizer.vocabulary_)
#
# print(x.toarray())
import json
import os
import re

import numpy as np
from gensim.models import Word2Vec

from gumtree import SubTreeExtractor

# import os
#
# BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# data_path = os.path.join(BASE_PATH, 'data/apache')

# # 读取CSV文件
# df = pd.read_csv('../data/apache_test_large.csv')

# # 获取所有唯一的project值
# projects = df['project'].unique()
#
# # 按照project列的值拆分并保存为多个CSV文件
# for project in projects:
#
#     name = project.split('/')[1]
#
#     # 筛选出当前project的所有行
#     df_project = df[df['project'] == project]
#
#     # 生成文件名
#     file_name = os.path.join(data_path, 'test/' + f'{name}.csv')
#
#     # 保存为新的CSV文件
#     df_project.to_csv(file_name, index=False)
#     print(f'Saved: {file_name}')

# gr = Git('~/Desktop/repos/groovy')
# commit = gr.get_commit('7b8480744ea6e6fb41efd4329bb470c8f3c763db')
# print(commit.author_date.timestamp())
#
BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# commit_folder = os.path.join(BASE_PATH, 'data')
data_path = os.path.join(BASE_PATH, 'data')

# dot_path = os.path.join(BASE_PATH, 't.dot')
# dotfiles = {
#     'before': [],
#     'after': []
# }
# current = 'before'
# node_pattern = '^n_[a-z]+_[0-9]+ \\[label=".+", color=(red|blue|green|orange|lightgrey)\\];$'
# edge_pattern = '^n_[a-z]+_[0-9]+ -> n_[a-z]+_[0-9]+;$'
# cnt = 0
# with open(dot_path, 'r') as f:
#     for l in f:
#         l = l.strip()
#         if l == 'subgraph cluster_dst {':
#             current = 'after'
#         if not re.match(node_pattern, l) and not re.match(edge_pattern, l):
#             cnt = cnt + 1
#             continue
#         dotfiles[current].append(l)
#
# print(cnt)
# # print(dotfiles['before'])
# extractor = SubTreeExtractor(dotfiles['after'])
# features, edges, colors = extractor.extract_subtree()
# # print(features)
# extractor.generate_dotfile()


special_token = False
files = ['/camel_train_1.json', '/camel_val_1.json']
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

# for wd in corpus:
#     print(wd)
corpus.append(['<UNK>'])
vectorizer_model = Word2Vec(corpus, vector_size=100, sg=0, window=5, min_count=1, workers=4)
words = list(vectorizer_model.wv.key_to_index.keys())
print(len(words))
print(vectorizer_model.vector_size)
# for word in words[:10]:
#     print(word)
# print(vectorizer_model.wv['<UNK>'])
# print(vectorizer_model.wv['Insert_SimpleName: createInitialContext'])

# b_node_tokens, b_edges, b_colors = [], [[], []], []
# a_node_tokens, a_edges, a_colors = [], [[], []], []
# b_nodes_so_far, a_nodes_so_far = 0, 0
# for f_name in files:
#     with open(data_path + f_name) as fp:
#         subtrees = json.load(fp)
#     for commit, files in subtrees.items():
#         for file in files:
#             b_node_tokens += [' '.join(node) for node in file[1][0]]
#             b_colors += [c for c in file[1][2]]
#             b_edges = [
#                 b_edges[0] + [s + b_nodes_so_far for s in file[1][1][0]],  # source nodes
#                 b_edges[1] + [d + b_nodes_so_far for d in file[1][1][1]]  # destination nodes
#             ]
#             a_node_tokens += [' '.join(node) for node in file[2][0]]
#             a_colors += [c for c in file[2][2]]
#             a_edges = [
#                 a_edges[0] + [s + a_nodes_so_far for s in file[2][1][0]],  # source nodes
#                 a_edges[1] + [d + a_nodes_so_far for d in file[2][1][1]]  # destination nodes
#             ]
#
# print(b_node_tokens)
# def get_embedding_from_word2vec(node_features, model):
#     embeddings = []
#     for feature in node_features:
#         if feature in model.wv:
#             embeddings.append(model.wv[feature])  # 获取词向量
#         else:
#             embeddings.append(np.zeros(model.vector_size))  # 未知特征使用零向量
#     return np.mean(embeddings, axis=0)  # 对所有节点的词向量取平均
