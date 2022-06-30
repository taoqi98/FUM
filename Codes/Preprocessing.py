from Hypers import *
from Utils import *
from nltk.tokenize import word_tokenize
import json

def read_news(data_root_path):
    
    news={}

    news_index={}
    index=1

    word_dict={}
    word_index=1
    
    content_count = {}
    content_dict = {}
    content_index = 1

    entity_dict = {}
    entity_index = 1

    category_dict={}
    category_index = 1

    subcategory_dict={}
    subcategory_index = 1

    for path in ['MINDlarge_train','MINDlarge_test']:

        with open(os.path.join(data_root_path,path,'news.tsv')) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip('\n').split('\t')
            doc_id,vert,subvert,title,abstract,url,entity,_= splited
            entity = json.loads(entity)
            if doc_id in news_index:
                continue

            news_index[doc_id]=index
            index+=1

            title = word_tokenize(title.lower())  
            abstract = abstract.lower().split()[:MAX_CONTENT]
            entity = [e['WikidataId'] for e in entity]

            news[doc_id]=[vert,subvert,title,entity,abstract]

            for word in title:
                if not(word in word_dict):
                    word_dict[word]=word_index
                    word_index+=1
                    
            for word in abstract:
                if not (word in content_count):
                    content_count[word] = 0
                content_count[word] += 1

            for e in entity:
                if not (e in entity_dict):
                    entity_dict[e] = entity_index
                    entity_index += 1

            if not vert in category_dict:
                category_dict[vert] = category_index
                category_index += 1

            if not subvert in subcategory_dict:
                subcategory_dict[subvert] = subcategory_index
                subcategory_index += 1
                
    for word in content_count:
        if content_count[word]<3:
            continue
        content_dict[word] = content_index
        content_index += 1
        
    return news,news_index,category_dict,subcategory_dict,word_dict,content_dict,entity_dict



def get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict,content_dict,entity_dict):
    news_num=len(news)+1
    news_title=np.zeros((news_num,MAX_TITLE),dtype='int32')
    news_vert=np.zeros((news_num,),dtype='int32')
    news_subvert=np.zeros((news_num,),dtype='int32')
    news_entity = np.zeros((news_num,MAX_ENTITY),dtype='int32')
    news_content = np.zeros((news_num,MAX_CONTENT),dtype='int32')
    
    for key in news:    
        vert,subvert,title,entity,content = news[key]
        doc_index=news_index[key]
        
        news_vert[doc_index]=category_dict[vert]
        news_subvert[doc_index]=subcategory_dict[subvert]
        
        for word_id in range(min(MAX_TITLE,len(title))):
            news_title[doc_index,word_id]=word_dict[title[word_id]]
        
        for entity_id in range(min(MAX_ENTITY,len(entity))):
            news_entity[doc_index,entity_id]=entity_dict[entity[entity_id]]

        for content_id in range(min(MAX_ENTITY,len(content))):
            if not content[content_id] in content_dict:
                continue
            news_content[doc_index,content_id]=content_dict[content[content_id]]

    return news_title,news_vert,news_subvert,news_entity,news_content

def read_train_clickhistory(news_index,data_root_path,filename):
    
    lines = []
    with open(os.path.join(data_root_path,filename)) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions

def read_test_clickhistory(news_index,data_root_path,filename):
    
    lines = []
    with open(os.path.join(data_root_path,filename)) as f:
        for i in range(200000):
            l = f.readline()
            lines.append(l)
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions


def read_test_clickhistory_noclk(news_index,data_root_path,filename):
    
    lines = []
    with open(os.path.join(data_root_path,filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clicks = []
        else:
            clicks = click.split()
        true_click = []
        for j in range(len(clicks)):
            click = clicks[j]
            assert click in news_index
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            pos.append(imp)
        sessions.append([true_click,pos,neg])
    return sessions


def parse_user(news_index,session):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_CLICK),dtype='int32'),}
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg =session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) >MAX_CLICK:
            click = click[-MAX_CLICK:]
        else:
            click=[0]*(MAX_CLICK-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user


def get_train_input(news_index,session):
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)

    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    
    return sess_all, user_id, label

def get_test_input(news_index,session):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,