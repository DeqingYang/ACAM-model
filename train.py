from metrics import prec_score, ap_score, ndcg_score
import numpy as np
import torch
from model import KAAM
from torch.nn import functional as F
from operator import itemgetter


def partitioning(lists):
    lists_1, lists_2 = lists[0:50], lists[50:100]
    num_users = int(lists.shape[0] / 50)
    data_index = 1
    for i in range(2, num_users):
        if(data_index % 10) : lists_2 = torch.cat((lists_1, lists[i * 50: (i + 1) * 50]))
        else : lists_1 = torch.cat((lists_2, lists[i * 50: (i + 1) * 50]))
        data_index += 1
    print(len(lists_1))
    return lists_1, lists_2

def test(net, test_seq):
    k_index = [3, 5, 10]
    net.eval()
    label = np.array([1, 0, 0, 0, 0] * 10)
    prec, ap, ndcg, rr_list = [[], [], []], [[], [], []], [[], [], []], []
    num_users = int(test_seq.shape[0] / 50)
    for i in range(num_users):
        score = F.softmax(net(test_seq[i * 50: (i + 1) * 50])[0], dim=1)[:, 1].cpu().detach().numpy()
        ordered = sorted(zip(label, score), key=itemgetter(1), reverse=True)
        ordered_label = [i[0] for i in ordered]
        for (i, k) in zip(range(0, 3), k_index):
            prec[i].append(prec_score(ordered_label, k))
            ap[i].append(ap_score(ordered_label, k))
            ndcg[i].append(ndcg_score(ordered_label, k))
        rr_list.append(1 + ordered_label.index(1))
    rr = np.mean(1 / np.array(rr_list))
    result = [np.mean(v) for v in prec] + [np.mean(v) for v in ap] + [np.mean(v) for v in ndcg] + [rr]
    net.train()
    return result


def train(args, loader, train_set_length, test_seq, item_attr_set, ent_embed, attr_embed):
    test_seq, valid_seq = partitioning(test_seq)

    if(args.method=='KAAM'):
        net = KAAM(args, ent_embed, attr_embed, item_attr_set, loader, use_user_attn=True,use_item_attn=True, use_pretrain=True)
    if(args.method=='AAM'):
        net = KAAM(args, [], [], item_attr_set, loader, use_user_attn=True,use_item_attn=True, use_pretrain=False)

    # net._initialize_weights()
    prec3_max=0
    for k in range(0, args.epoch):
        print(k)
        net.calculate(args, train_set_length)
        if((k%3 >= 0)):
            result_temp = test(net, test_seq)
            if (result_temp[0] > prec3_max):
                print('epoch=', k, '\nresult on test data:', result_temp, '\n')
                prec3_max = result_temp[0]
                result_valid = test(net, valid_seq)
                print('result on valid data:', result_valid)
        if (k % 6 == 5):
            net.learning_rate_adjust(args)