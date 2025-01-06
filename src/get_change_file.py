import os

import pandas as pd
from pydriller import Git

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')
# MAX_N_CHANGED_FILES = 3


if __name__ == "__main__":

    df = pd.read_csv(os.path.join(data_path, 'apache_total.csv'))

    for index, row in df.iterrows():
        if index < 1:
            cmt_id = row['commit_id']
            project = row['project']
            repo = project.split('/')[-1]
            print(f'Processing commit {cmt_id} from repository {repo}')

            # 创建项目文件夹
            project_folder = os.path.join(data_path, repo)
            os.makedirs(project_folder, exist_ok=True)

            # 创建提交文件夹
            commit_folder = os.path.join(project_folder, cmt_id)
            os.makedirs(commit_folder, exist_ok=True)

            gr = Git(os.path.join('~/Desktop/repos', repo))
            commit = gr.get_commit(cmt_id)

            # for commit in Repository(repo, single=cmt_id).traverse_commits():
            for mod in commit.modified_files:

                if not mod.filename.endswith('.java'):
                    continue

                mod_file_before = os.path.join(commit_folder,
                                               mod.filename.split('.')[0] + '_b.' + mod.filename.split('.')[1])
                mod_file_after = os.path.join(commit_folder,
                                              mod.filename.split('.')[0] + '_a.' + mod.filename.split('.')[1])

                # 获取修改前的源文件内容（如果文件存在）
                if mod.source_code_before:
                    with open(mod_file_before, 'w', encoding='utf-8') as f:
                        f.write(mod.source_code_before)
                    print(f'Saved before file to {mod_file_before}')

                # 获取修改后的源文件内容
                if mod.source_code:
                    with open(mod_file_after, 'w', encoding='utf-8') as f:
                        f.write(mod.source_code)
                    print(f'Saved after file to {mod_file_after}')

            print('\n')
