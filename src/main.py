import os
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader

from models import JITGNN
from datasets import ASTDataset
from train import pretrain, test, resume_training, plot_training, train
import argparse

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epoch", default=15, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--message_size", default=32, type=int)
    parser.add_argument("--metric_size", default=13, type=int)
    parser.add_argument("--word2vec_dim", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    args = parser.parse_args()

    train_dataset = ASTDataset(os.path.join(data_path, 'camel_train.h5'))
    val_dataset = ASTDataset(os.path.join(data_path, 'camel_val.h5'))
    test_dataset = ASTDataset(os.path.join(data_path, 'camel_test.h5'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    hidden_size = args.word2vec_dim + 2  # plus supernode node feature and node colors
    print('hidden_size is {}'.format(hidden_size))

    model = JITGNN(hidden_size, args.message_size, args.metric_size)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training
    pretrain(model, optimizer, criterion, args.epoch, train_loader, val_loader)
    # train_features = torch.load(os.path.join(BASE_PATH, 'trained_models/train_features.pt')).cpu().detach().numpy()
    # train_labels = torch.load(os.path.join(BASE_PATH, 'trained_models/train_labels.pt')).cpu().detach().numpy()
    # clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    # train(clf, train_features, train_labels)

    if args.test:
        model = torch.load(os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
        test(model, test_loader)
        # test(model, dataset)
        # test(model, dataset)
