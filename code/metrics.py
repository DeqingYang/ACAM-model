# -*- coding: utf-8 -*-
import numpy as np


def prec_score(y_true, k):
    prec = sum(y_true[:k]) / k
    return prec
    

def ndcg_score(y_true, k):
    idcg = sum([1/np.log2(i + 1) for i in np.arange(1, k + 1)])
    dcg = sum((2**np.array(y_true)[:k] - 1) / np.log2(np.arange(1, k + 1) + 1))  
    return dcg / idcg


def ap_score(y_true, k):
    p = []
    for i in range(1, k + 1):
        p.append(sum(y_true[:i]) / i)
    ap = sum(np.array(p) * y_true[:k]) / k 
    return ap
