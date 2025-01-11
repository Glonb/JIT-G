import json
import logging
import math
import os
import re
import subprocess
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
from pydriller import Git
from tqdm import tqdm

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class GumTreeDiff:
    def __init__(self):
        self.bin_path = os.path.join(BASE_PATH, 'gumtree-4.0.0/bin/gumtree')
        self.src_dir = os.path.join(data_path, 'src')
        if not os.path.exists(self.src_dir):
            os.makedirs(self.src_dir)

    def get_diff(self, file_name, b_content, a_content):
        f_name = file_name.split('/')[-1]
        b_file = os.path.join(self.src_dir, f_name.split('.')[0] + '_b.' + f_name.split('.')[1])
        a_file = os.path.join(self.src_dir, f_name.split('.')[0] + '_a.' + f_name.split('.')[1])
        with open(b_file, 'w') as file:
            file.write(b_content)
        with open(a_file, 'w') as file:
            file.write(a_content)
        command = subprocess.Popen([self.bin_path, 'dotdiff', b_file, a_file],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        output, error = command.communicate()
        if error.decode('utf-8'):
            return None
        return output.decode('utf-8')

    def get_dotfiles(self, file):
        dot = self.get_diff(file[0], file[1], file[2])
        if dot is None:
            raise SyntaxError()
        lines = dot.splitlines()
        dotfiles = {
            'before': [],
            'after': []
        }
        current = 'before'
        node_pattern = '^n_[a-z]+_[0-9]+ \\[label=".+", color=(red|blue|green|orange|lightgrey)\\];$'
        edge_pattern = '^n_[a-z]+_[0-9]+ -> n_[a-z]+_[0-9]+;$'
        for l in lines:
            l = l.lstrip()
            if l == 'subgraph cluster_dst {':
                current = 'after'
            if not re.match(node_pattern, l) and not re.match(edge_pattern, l):
                continue
            dotfiles[current].append(l)
        return dotfiles['before'], dotfiles['after']


class SubTreeExtractor:
    def __init__(self, dot):
        self.dot = dot
        self.red_nodes = list()
        self.green_nodes = list()  # 新增绿色节点列表
        self.orange_nodes = list()  # 新增橙色节点列表
        self.blue_nodes = list()  # 新增蓝色节点列表
        self.node_dict = dict()
        self.from_to = dict()  # mapping from src nodes to list of their dst nodes.
        self.to_from = dict()  # mapping from dst nodes to list of their src nodes.
        self.subtree_nodes = set()
        self.subtree_edges = set()

    def read_ast(self):
        node_pattern = '^n_[a-z]+_[0-9]+ \\[label=".+", color=(red|blue|green|orange|lightgrey)\\];$'
        edge_pattern = '^n_[a-z]+_[0-9]+ -> n_[a-z]+_[0-9]+;$'
        for line in self.dot:
            if re.match(node_pattern, line):
                assert len(re.findall('^(.*) \\[label=".+", color=.+\\];$', line)) == 1
                id = re.findall('^(.*) \\[label=".+", color=.+\\];$', line)[0]
                assert len(re.findall('^n_[a-z]+_[0-9]+ \\[label="(.+)", color=.+\\];$', line)) == 1
                unclean_label = re.findall('^n_[a-z]+_[0-9]+ \\[label="(.+)", color=.+\\];$', line)[0]
                label = re.split('\\[[0-9]+', unclean_label)[0]
                # 去掉不完整的 "[数字" 或 "[数字,后续内容" 格式
                label = re.sub(r'\[\d*(?:,.*)?$', '', label).strip()
                assert len(re.findall('color=(red|blue|green|orange|lightgrey)\\];$', line)) == 1
                color = re.findall('color=(red|blue|green|orange|lightgrey)\\];$', line)[0]

                self.node_dict[id] = label
                if color == 'red':
                    self.red_nodes.append(id)
                elif color == 'green':
                    self.green_nodes.append(id)  # 处理绿色节点
                elif color == 'orange':
                    self.orange_nodes.append(id)  # 处理橙色节点
                elif color == 'blue':
                    self.blue_nodes.append(id)  # 处理蓝色节点

            elif re.match(edge_pattern, line):
                assert len(re.findall('^(.*) -> n_[a-z]+_[0-9]+;$', line)) == 1
                source = re.findall('^(.*) -> n_[a-z]+_[0-9]+;$', line)[0]
                assert len(re.findall('^n_[a-z]+_[0-9]+ -> (.*);$', line)) == 1
                dest = re.findall('^n_[a-z]+_[0-9]+ -> (.*);$', line)[0]

                if source not in self.from_to:
                    self.from_to[source] = [dest]
                else:
                    self.from_to[source].append(dest)
                if dest not in self.to_from:
                    self.to_from[dest] = [source]
                else:
                    self.to_from[dest].append(source)

            else:
                print(line, end='\t')

    def extract_subtree(self):
        self.read_ast()

        # 提取红色、绿色、橙色节点相关的子树
        for n in self.red_nodes + self.green_nodes + self.orange_nodes:
            # for n in self.red_nodes:
            self.subtree_nodes.add(n)
            if n in self.from_to:
                for d in self.from_to[n]:
                    self.subtree_nodes.add(d)
                    self.subtree_edges.add((n, d))
            if n in self.to_from:
                for s in self.to_from[n]:
                    self.subtree_nodes.add(s)
                    self.subtree_edges.add((s, n))
                    for d in self.from_to[s]:
                        self.subtree_nodes.add(d)
                        self.subtree_edges.add((s, d))

        vs = list(self.subtree_nodes)
        es = list(self.subtree_edges)
        colors = ['red' if node_id in self.red_nodes else
                  'green' if node_id in self.green_nodes else
                  'orange' if node_id in self.orange_nodes else
                  'blue' if node_id in self.blue_nodes else
                  'lightgrey' for node_id in vs]

        # 根据节点颜色为features中的元素加上不同的标志
        features = []
        for node_id in vs:
            label = self.node_dict.get(node_id, 'unknown')

            # 根据颜色添加标志
            if node_id in self.red_nodes:
                features.append([f"Delete_{label}"])
            elif node_id in self.green_nodes:
                features.append([f"Insert_{label}"])
            elif node_id in self.orange_nodes:
                features.append([f"Update_{label}"])
            elif node_id in self.blue_nodes:
                features.append([f"Move_{label}"])
            else:
                features.append([label])

        # features = [[self.node_dict[node_id]] if node_id in self.node_dict else ['unknown'] for node_id in vs]
        edges = [[], []]
        for src, dst in es:
            edges[0].append(vs.index(src))
            edges[1].append(vs.index(dst))

        return features, edges, colors

    def generate_dotfile(self):
        content = 'digraph G {\nnode [style=filled];\nsubgraph cluster_dst {\n'
        for node in self.subtree_nodes:
            # 获取节点的标签，默认设置为 "unknown"
            label = self.node_dict.get(node, 'unknown')

            # 根据节点颜色确定其颜色
            if node in self.red_nodes:
                color = 'red'
            elif node in self.green_nodes:
                color = 'green'
            elif node in self.orange_nodes:
                color = 'orange'
            elif node in self.blue_nodes:
                color = 'blue'
            else:
                color = 'lightgrey'  # 如果节点不在红色、绿色、橙色或蓝色节点中，则默认为亮灰色

            # 添加节点信息到content
            content += '{} [label="{}", color={}];\n'.format(node, label, color)

            # content += '{} [label="{}", color={}];\n'.format(node,
            #                                                  self.node_dict[node],
            #                                                  'blue' if node not in self.red_nodes else 'red')
        for edge in self.subtree_edges:
            content += '{} -> {};\n'.format(edge[0], edge[1])
        content += '}\n;}\n'

        with open(os.path.join(data_path, 'src', 'new.dot'), 'w') as file:
            file.write(content)


class RunHandler:
    def __init__(self, commit_file, ast_filename, already_file, types, limit=10000):
        self.commit_file = commit_file
        self.ast_filename = ast_filename
        self.types = types
        self.already_file = already_file
        self.limit = limit
        self.ast_dict = dict()
        self.file_index = 1
        self.save_file = None
        self.already = []
        self.commits = dict()
        self.initialize()
        Path("logs/").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                RotatingFileHandler(filename='logs/pydriller.log', maxBytes=5 * 1024 * 1024,
                                                    backupCount=5)])

    @staticmethod
    def time_since(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '{} min {:.2f} sec'.format(m, s)

    def initialize(self):
        while not self.save_file:
            filepath = os.path.join(data_path, self.ast_filename + '_{}.json'.format(self.file_index))
            if os.path.isfile(filepath):
                with open(filepath) as file:
                    self.ast_dict = json.load(file)
                if len(self.ast_dict) == self.limit:
                    self.file_index += 1
                    continue
                self.save_file = filepath
            else:
                self.save_file = filepath
        print('current file: {}\nast dict size: {}\n'.format(self.save_file, len(self.ast_dict)))
        self.already = pd.read_csv(os.path.join(data_path, self.already_file))['commit_id'].tolist()
        df = pd.read_csv(os.path.join(data_path, self.commit_file))
        commits = df['commit_id']
        projects = df['project']
        commits = dict(zip(commits, projects))
        remaining = list(set(commits.keys()) - set(self.already) - set(self.ast_dict.keys()))
        self.commits = {k: commits[k] for k in remaining}

    def has_modification_with_file_type(self, commit):
        for mod in commit.modified_files:
            if mod.filename.endswith(tuple(self.types)):
                return True
        return False

    def is_filtered(self, commit):
        # if commit.files > 100:
        #     logging.info('Too many files.')
        #     return True
        # if commit.lines > 10000:
        #     logging.info('Too many lines.')
        #     return True
        if not self.has_modification_with_file_type(commit):
            logging.info('No file in given language')
            return True
        return False

    def filter_commits(self):
        """
        filter commits based on PyDriller
        """
        filtered, projects, dates = [], [], []
        for c, p in tqdm(self.commits.items(), desc='filtering commits', unit='commits'):
            try:
                commit = Git('../repos/' + p.split('/')[1]).get_commit(c)
            except ValueError:  # for hadoop repos
                commit = Git('../repos/' + p.split('/')[1].split('-')[0]).get_commit(c)
            logging.info('Commit #%s in %s from %s', commit.hash, commit.committer_date, commit.author.name)
            if self.is_filtered(commit):
                continue
            else:
                filtered.append(c)
                projects.append(p)
                dates.append(int(commit.committer_date.timestamp()))
                if len(filtered) % 1000 == 0:
                    pd.DataFrame({'commit_id': filtered, 'project': projects, 'date': dates}) \
                        .to_csv(os.path.join(data_path, 'filtered_commits.csv'), index=False)
        pd.DataFrame({'commit_id': filtered, 'project': projects, 'date': dates}) \
            .to_csv(os.path.join(data_path, 'filtered_commits.csv'), index=False)

    def store_subtrees(self):
        gumtree = GumTreeDiff()
        dataset_start = time.time()
        for c, p in tqdm(self.commits.items(), desc='storing subtrees', unit='commits'):
            try:
                commit = Git('../repos/' + p.split('/')[1]).get_commit(c)
            except ValueError:  # for hadoop repos
                commit = Git('../repos/' + p.split('/')[1].split('-')[0]).get_commit(c)
            logging.info('Commit #%s in %s from %s', commit.hash, commit.committer_date, commit.author.name)
            commit_start = time.time()
            for m in commit.modified_files:
                if not m.filename.endswith(tuple(self.types)):
                    continue
                filepath = m.new_path if m.new_path is not None else m.old_path
                before = m.source_code_before if m.source_code_before is not None else ''
                after = m.source_code if m.source_code is not None else ''
                f = (filepath, before, after)
                try:
                    b_dot, a_dot = gumtree.get_dotfiles(f)
                except SyntaxError:
                    print('\t\t\t\tsource code has syntax error. PASS!')
                    continue
                subtree = SubTreeExtractor(b_dot)
                b_subtree = subtree.extract_subtree()
                subtree = SubTreeExtractor(a_dot)
                a_subtree = subtree.extract_subtree()

                # to exclude ast with no red nodes (which have empty subtrees)
                # this includes F1 in McIntosh & Kamei (comment and whitespace filtering)
                if len(b_subtree[0]) == 0 and len(a_subtree[0]) == 0:
                    # print("Bad !!!!!!!!!!!!!!!!!")
                    continue
                elif len(b_subtree[0]) == 0:
                    b_subtree[0].append(['None'])
                elif len(a_subtree[0]) == 0:
                    a_subtree[0].append(['None'])

                if commit.hash not in self.ast_dict:
                    self.ast_dict[commit.hash] = [(filepath, b_subtree, a_subtree)]
                else:
                    self.ast_dict[commit.hash].append((f[0], b_subtree, a_subtree))

            if commit.hash in self.ast_dict:  # to check if new commit is added
                print('\ncommit {} subtrees collected in {}.'.format(commit.hash[:7], self.time_since(commit_start)))
                if len(self.ast_dict) % 100 == 0:
                    with open(self.save_file, 'w') as fp:
                        json.dump(self.ast_dict, fp)
                    print('\n\n***** ast_dict backup saved at size {}. *****\n\n'.format(len(self.ast_dict)))
                    if len(self.ast_dict) == self.limit:
                        self.file_index += 1
                        self.save_file = os.path.join(data_path, self.ast_filename + '_{}.json'.format(self.file_index))
                        self.already += list(self.ast_dict.keys())
                        pd.DataFrame({'commit_id': self.already}) \
                            .to_csv(os.path.join(data_path, self.already_file), index=False)
                        self.ast_dict = dict()
                        print('\n\n***** switching file *****\n\n')

        print('\nall {} commit trees extracted in {}'.format(len(self.commits), self.time_since(dataset_start)))
        with open(self.save_file, 'w') as fp:
            json.dump(self.ast_dict, fp)
        self.already += list(self.ast_dict.keys())
        pd.DataFrame({'commit_id': self.already}) \
            .to_csv(os.path.join(data_path, self.already_file), index=False)


if __name__ == '__main__':
    # RunHandler(commit_file='camel.csv',
    #            ast_filename='camel_subtrees',
    #            already_file='already_filtered.csv',
    #            types=['.java']).filter_commits()
    RunHandler(commit_file='camel_test.csv',
               ast_filename='camel_test',
               already_file='already_filtered.csv',
               types=['.java']).store_subtrees()
    # subtree = SubTreeExtractor()
    # subtree.read_ast()
    # subtree.extract_subtree()
    # subtree.generate_dotfile()
    print('finished.')
