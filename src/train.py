import math
import os
import time
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from scipy.optimize import differential_evolution
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import pandas as pd
from metrics import roc_auc, calculate_metrics
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

cloud_path = '/root/autodl-tmp'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


def time_since(since):
    now = time.time()
    s = now - since
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '{}h {}min {:.2f} sec'.format(h, m, s)


# def evaluate(label, output):
#     return roc_auc(np.array(label), np.array(output))


def evaluate(label, output):
    # 如果 label 和 output 是 Python 列表，先将它们转换为 PyTorch 张量
    if isinstance(label, list):
        label = torch.tensor(label)
    if isinstance(output, list):
        output = torch.tensor(output)

    # 确保 label 和 output 都在 CPU 上，然后转换为 NumPy 数组
    label = label.cpu().numpy()  # 将 label 移动到 CPU 并转换为 NumPy 数组
    output = output.cpu().numpy()  # 将 output 移动到 CPU 并转换为 NumPy 数组

    return roc_auc(label, output)


# def evaluate_more(label, output):
#     return calculate_metrics(np.array(label), np.array(output))

def evaluate_more(label, output):
    # 如果 label 和 output 是 Python 列表，先将它们转换为 PyTorch 张量
    if isinstance(label, list):
        label = torch.tensor(label)
    if isinstance(output, list):
        output = torch.tensor(output)

    # 确保 label 和 output 都在 CPU 上，然后转换为 NumPy 数组
    label = label.cpu().numpy()
    output = output.cpu().numpy()

    return calculate_metrics(label, output)


def pretrain(model, optimizer, criterion, epochs, train_loader, val_loader, so_far=0, resume=None):
    if resume:
        all_training_aucs = resume['all_training_aucs']
        all_training_losses = resume['all_training_losses']
        all_val_aucs = resume['all_val_aucs']
        all_val_losses = resume['all_val_losses']
    else:
        all_training_aucs = []
        all_training_losses = []
        all_val_aucs = []
        all_val_losses = []

    print('--------training--------')
    model = model.to(device)
    for e in range(epochs):
        print('epoch {:3d}/{}'.format((e + 1 + so_far), (epochs + so_far)))
        # training
        start = time.time()
        total_loss = 0
        y_scores = []
        y_true = []

        model.train()
        i = 0
        for data in train_loader:
            if data is None:
                continue

            before_embeddings, before_adj, after_embeddings, after_adj, label, metric = data
            optimizer.zero_grad()

            # 将数据移动到GPU
            before_embeddings = before_embeddings.to(device)
            before_adj = before_adj.to(device)
            after_embeddings = after_embeddings.to(device)
            after_adj = after_adj.to(device)
            label = label.to(device)
            metric = metric.to(device)

            output, features = model(before_embeddings, before_adj,
                                     after_embeddings, after_adj,
                                     metric)

            loss = criterion(output, torch.Tensor([label]).to(device))
            loss.backward()
            optimizer.step()

            y_scores.append(torch.sigmoid(output).item())
            y_true.append(label)

            total_loss += loss.item()
            i += 1
            if i % 100 == 0:
                print('[{:5d}]\tloss: {:.4f}'.format(i, loss.item()))

        print('epoch duration: {}'.format(time_since(start)))

        # torch.save({
        #     'epoch': e + 1 + so_far,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()
        # }, os.path.join(BASE_PATH, 'trained_models/checkpoint.pt'))
        # print('* checkpoint saved.')

        training_loss = total_loss / i
        _, _, _, training_auc = evaluate(y_true, y_scores)
        precision, recall, f1, mcc = evaluate_more(y_true, y_scores)
        print('\n<==== training loss = {:.4f} ====>'.format(training_loss))
        print('metrics: AUC={}\tPrecision={}\t'
              'Recall={}\tF1={}\tMCC={}'.format(training_auc, precision, recall, f1, mcc))

        all_training_losses.append(training_loss)
        all_training_aucs.append(training_auc)

        # validation
        total_loss = 0
        y_scores = []
        y_true = []

        print('--------validation--------')
        model.eval()
        i = 0
        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue

                before_embeddings, before_adj, after_embeddings, after_adj, label, metric = data

                # 将数据移动到GPU
                before_embeddings = before_embeddings.to(device)
                before_adj = before_adj.to(device)
                after_embeddings = after_embeddings.to(device)
                after_adj = after_adj.to(device)
                label = label.to(device)
                metric = metric.to(device)

                output, features = model(before_embeddings, before_adj,
                                         after_embeddings, after_adj,
                                         metric)

                loss = criterion(output, torch.Tensor([label]).to(device))
                total_loss += loss.item()

                y_scores.append(torch.sigmoid(output).item())
                y_true.append(label)

                i += 1

        val_loss = total_loss / i
        _, _, _, val_auc = evaluate(y_true, y_scores)
        precision, recall, f1, mcc = evaluate_more(y_true, y_scores)
        print('<==== validation loss = {:.4f} ====>'.format(val_loss))
        print('metrics: AUC={}\tPrecision={}\t'
              'Recall={}\tF1={}\tMCC={}\n'.format(val_auc, precision, recall, f1, mcc))

        if len(all_val_aucs) == 0 or val_auc > max(all_val_aucs):
            torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
            print('* model_best_auc saved.')

        if len(all_val_losses) == 0 or val_loss < min(all_val_losses):
            torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_least_loss.pt'))
            print('* model_least_loss saved.')

        all_val_losses.append(val_loss)
        all_val_aucs.append(val_auc)

        torch.save({
            'all_training_losses': all_training_losses,
            'all_training_aucs': all_training_aucs,
            'all_val_losses': all_val_losses,
            'all_val_aucs': all_val_aucs,
        }, os.path.join(BASE_PATH, 'trained_models/stats.pt'))
        print('* stats saved.')

    torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_final.pt'))
    print('* model_final saved.')
    print('\ntraining finished')


def objective_func(k, train_features, train_labels, valid_features, valid_labels):
    smote = SMOTE(random_state=42, k_neighbors=int(np.round(k)), n_jobs=32)
    train_feature_res, train_label_res = smote.fit_resample(train_features, train_labels)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(train_feature_res, train_label_res)
    prob = clf.predict_proba(valid_features)[:, 1]
    auc = roc_auc_score(valid_labels, prob)

    return -auc


def train(clf, train_features, train_labels):
    percent_80 = int(train_features.shape[0] * 0.8)
    train_features, valid_features = train_features[:percent_80], train_features[percent_80:]
    train_labels, valid_labels = train_labels[:percent_80], train_labels[percent_80:]
    bounds = [(1, 20)]
    opt = differential_evolution(objective_func, bounds, args=(train_features, train_labels,
                                                               valid_features, valid_labels),
                                 popsize=10, mutation=0.7, recombination=0.3, seed=0)
    smote = SMOTE(random_state=42, n_jobs=32, k_neighbors=int(np.round(opt.x)))
    train_features, train_labels = smote.fit_resample(train_features, train_labels)
    clf.fit(train_features, train_labels)
    prob = clf.predict_proba(valid_features)[:, 1]
    _, _, _, auc = evaluate(valid_labels, prob)
    print('metrics: AUC={}\n'.format(auc))


def test(model, test_loader):
    print('--------testing--------')
    y_scores = []
    y_true = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            if data is None:
                continue

            before_embeddings, before_adj, after_embeddings, after_adj, label, metric = data

            # 将数据移动到GPU
            before_embeddings = before_embeddings.to(device)
            before_adj = before_adj.to(device)
            after_embeddings = after_embeddings.to(device)
            after_adj = after_adj.to(device)
            label = label.to(device)
            metric = metric.to(device)

            output, features = model(before_embeddings, before_adj,
                                     after_embeddings, after_adj,
                                     metric)

            y_scores.append(torch.sigmoid(output))
            y_true.append(label)

        # 合并所有 batch 的结果，并将其转换为 PyTorch 张量
    y_scores = torch.cat(y_scores, dim=0)
    y_true = torch.cat(y_true, dim=0)

    # 确保 y_true 和 y_scores 都在相同的设备上（CPU 或 GPU）
    y_scores = y_scores.to(device)
    y_true = y_true.to(device)

    # 将结果保存到 CSV 文件
    pd.DataFrame({'y_true': y_true.cpu().numpy(), 'y_score': y_scores.cpu().numpy()}).to_csv(
        os.path.join(data_path, 'test_result.csv'))

    # pd.DataFrame({'y_true': y_true, 'y_score': y_scores}).to_csv(os.path.join(data_path, 'test_result.csv'))
    fpr, tpr, thresholds, auc = evaluate(y_true, y_scores)
    precision, recall, f1, mcc = evaluate_more(y_true, y_scores)
    # print('metrics: AUC={}\n'.format(auc))
    print('metrics: AUC={}\tPrecision={}\t'
          'Recall={}\tF1={}\tMCC={}\n'.format(auc, precision, recall, f1, mcc))

    # print('thresholds: {}\n'.format(str(thresholds)))

    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/roc.png'))

    p, r, _ = precision_recall_curve(y_true, y_scores)
    plt.clf()
    plt.title('Precision-Recall')
    plt.plot(r, p, 'b', label='AUC = %0.2f' % metrics.auc(r, p))
    plt.legend(loc='upper right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/pr.png'))

    print('testing finished')


def resume_training(checkpoint, stats, model, optimizer, criterion, epochs, dataset):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    so_far = checkpoint['epoch']
    resume = {
        'all_training_losses': stats['all_training_losses'],
        'all_training_aucs': stats['all_training_aucs'],
        'all_val_losses': stats['all_val_losses'],
        'all_val_aucs': stats['all_val_aucs']
    }
    print('all set ...')
    pretrain(model, optimizer, criterion, epochs, dataset, so_far, resume)


def plot_training(stats):
    all_training_aucs = stats['all_training_aucs']
    all_training_losses = stats['all_training_losses']
    all_val_aucs = stats['all_val_aucs']
    all_val_losses = stats['all_val_losses']

    plt.figure()
    plt.plot(all_training_losses)
    plt.plot(all_val_losses)
    plt.title('Loss')
    plt.ylabel('Binary Cross Entropy')
    plt.xlabel('Epochs')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/loss.png'))

    plt.figure()
    plt.plot(all_training_aucs)
    plt.plot(all_val_aucs)
    plt.title('Performance')
    plt.ylabel('AUC')
    plt.xlabel('Epochs')
    plt.legend(['training auc', 'validation auc'], loc='lower right')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/performance.png'))


if __name__ == '__main__':
    print()
