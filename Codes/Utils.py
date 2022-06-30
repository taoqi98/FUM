import numpy as np
from sklearn.metrics import roc_auc_score
import os
from datetime import datetime
import time
import random
import json

def trans2tsp(timestr):
    return int(time.mktime(datetime.strptime(timestr, '%m/%d/%Y %I:%M:%S %p').timetuple()))

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

def shuffle(pn,labeler,pos):
    index=np.arange(pn.shape[0])
    pn=pn[index]
    labeler=labeler[index]
    pos=pos[index]
    
    for i in range(pn.shape[0]):
        index=np.arange(npratio+1)
        pn[i,:]=pn[i,index]
        labeler[i,:]=labeler[i,index]
    return pn,labeler,pos

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def load_matrix(embedding_path,word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1,300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix,have_word

def load_entity_embedding(data_root_path,entity_dict):
    entity_emb = np.zeros((len(entity_dict)+1,100))
    import pickle
    with open(os.path.join(data_root_path,'title_entity_emb.pkl'),'rb') as f:
        title_entity_emb = pickle.load(f)
    for eid in entity_dict:
        eix = entity_dict[eid]
        entity_emb[eix] = title_entity_emb[eid]
    return entity_emb

def evaluate(test_impressions,news_scoring,user_scoring):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        score = np.dot(nvs,uv)

        auc = roc_auc_score(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
        
    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()

    
    return AUC, MRR, nDCG5, nDCG10


def dump_result(test_imps,news_scoring,user_scoring,):
    result = []
    for i in range(len(test_imps)):
        
        nids = test_imps[i]['docs']

        uv = user_scoring[i]
        nvs = news_scoring[nids]
        scores = np.dot(nvs,uv)
        scores = (-scores).argsort()
        rank = np.zeros((len(scores),),dtype='int32')
        for j in range(len(scores)):
            rank[scores[j]] = j+1
        rank = json.dumps(rank.tolist())
        rank = ''.join(rank.split(' '))
        line = str(i+1) + ' '+ rank + '\n'
        result.append(line)
        
    with open('prediction.txt','w') as f:
        for i in range(len(result)):
            f.write(result[i])