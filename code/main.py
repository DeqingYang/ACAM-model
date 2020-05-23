import argparse
import numpy as np
from train import train
from dataset_build import load_data
import random
import torch
import os

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
# seed_torch(2)
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--L', type=int, default=3, help='How long is the records of user')
parser.add_argument('--epoch', type=int, default=300, help='how many observes of a epoch')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--l2_reg', type=float, default=0.0005, help='l2_reg')
parser.add_argument('--method', type=str, default="AAM", help='which model to be used')
parser.add_argument('--d', type=int, default=128, help='d')
parser.add_argument('--batch_size', type=int, default=3000, help='batch_size')
parser.add_argument('--use_KGloss', type=bool, default=True, help='use_KGloss')
parser.add_argument('--lambda_kg', type=float, default=0.0, help='lambda_kg')
parser.add_argument('--gamma', type=float, default=2, help='gamma')
parser.add_argument('--seed', type=float, default=0, help='gamma')

args = parser.parse_args()

loader, train_set_length, test_seq, item_attr_set = load_data(args)
print('Load data successful!')
train(args, loader, train_set_length, test_seq, item_attr_set, [], [])







