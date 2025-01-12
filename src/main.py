import os
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from models import JITGNN
from datasets import ASTDataset
from train import pretrain, test, resume_training, plot_training, train
import argparse

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epoch", default=15, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--message_size", default=32, type=int)
    args = parser.parse_args()

    # data_dict = {
    #     'train': ['/apache_train_50_all_1.json', '/apache_train_50_all_2.json',
    #               '/apache_train_50_all_3.json', '/apache_train_50_all_4.json'],
    #     'val': ['/apache_valid_50_all.json'],
    #     'test': ['/apache_test.json']
    # }
    # commit_lists = {
    #     'train': '/apache_train_50_all.csv',
    #     'val': '/apache_valid_50_all.csv',
    #     'test': '/apache_test.csv'
    # }
    data_dicts = {
        'train': ['/camel_train_1.json', '/camel_train_2.json'],
        'val': ['/camel_val_1.json'],
        'test': ['/camel_test_1.json'],
    }
    commit_lists = {
        'train': '/camel_train_filtered.csv',
        'val': '/camel_val_filtered.csv',
        'test': '/camel_test_filtered.csv'
    }

    dataset = ASTDataset(data_dicts, commit_lists, special_token=False)
    hidden_size = dataset.vectorizer_model.vector_size + 2  # plus supernode node feature and node colors
    metric_size = dataset.metrics.shape[1] - 1      # exclude commit_id column
    print('hidden_size is {}'.format(hidden_size))

    model = JITGNN(hidden_size, args.message_size, metric_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training
    pretrain(model, optimizer, criterion, args.epoch, dataset)
    # train_features = torch.load(os.path.join(BASE_PATH, 'trained_models/train_features.pt')).cpu().detach().numpy()
    # train_labels = torch.load(os.path.join(BASE_PATH, 'trained_models/train_labels.pt')).cpu().detach().numpy()
    # clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    # train(clf, train_features, train_labels)

    if args.test:
        model = torch.load(os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
        test(model, dataset)
        test(model, dataset)
        test(model, dataset)
