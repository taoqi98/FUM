import numpy as np
import keras
from keras.layers import Embedding
from keras.layers import *
from keras import backend as K
from keras.optimizers import *
from keras.models import Model
from keras.utils import multi_gpu_model

from Hypers import *


class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

class Fastformer(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        self.now_input_shape=None
        super(Fastformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.now_input_shape=input_shape
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True) 
        self.Wa = self.add_weight(name='Wa', 
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wb = self.add_weight(name='Wb', 
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)        
        self.WP = self.add_weight(name='WP', 
                                  shape=(self.output_dim,self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        
        super(Fastformer, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        
        
        Q_seq_D = K.reshape(Q_seq, (-1, self.now_input_shape[0][1], self.nb_head*self.size_per_head))
        
        Q_seq_A=  K.permute_dimensions(K.dot(Q_seq_D, self.Wa),(0,2,1))/ self.size_per_head**0.5
        if len(x)  == 5:
            Q_seq_A= Q_seq_A-(1-K.expand_dims(Q_len,axis=1))*1e8
        Q_seq_A =K.softmax(Q_seq_A)
        Q_seq = K.reshape(Q_seq, (-1,self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        
        
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1,self.now_input_shape[1][1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        
        Q_seq_AO=Q_seq_A
        Q_seq_A=Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=3),self.size_per_head,axis=3))(Q_seq_A)
        QA=K.sum(multiply([Q_seq_A, Q_seq]),axis=2)
        
        QA=Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=2), self.now_input_shape[1][1],axis=2))(QA)

        QAK=multiply([K_seq, QA])
        QAK_D = K.reshape(QAK, (-1, self.now_input_shape[0][1], self.nb_head*self.size_per_head))
        QAK_A = K.permute_dimensions(K.dot(QAK_D, self.Wb),(0,2,1))/ self.size_per_head**0.5
        if len(x)  == 5:
            QAK_A= QAK_A-(1-K.expand_dims(Q_len,axis=1))*1e8
        QAK_A=  K.softmax(QAK_A)
        
        QAK_AO=QAK_A
        QAK_A=Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=3),self.size_per_head,axis=3))(QAK_A)
        QK=K.sum(multiply([QAK_A, QAK]),axis=2)
        
        QKS=Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=2), self.now_input_shape[0][1],axis=2))(QK)
        
        QKQ=multiply([QKS, Q_seq])
        QKQ = K.permute_dimensions(QKQ, (0,2,1,3))
        QKQ=K.reshape(QKQ, (-1,self.now_input_shape[0][1], self.nb_head*self.size_per_head))
        QKQ=K.dot(QKQ, self.WP)
        QKQ=K.reshape(QKQ, (-1,self.now_input_shape[0][1], self.nb_head,self.size_per_head))
        QKQ = K.permute_dimensions(QKQ, (0,2,1,3))
        QKQ=QKQ+Q_seq
        QKQ = K.permute_dimensions(QKQ, (0,2,1,3))
        QKQ=K.reshape(QKQ, (-1,self.now_input_shape[0][1], self.nb_head*self.size_per_head))

        return QKQ
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model

def get_text_encoder(length,word_embedding_matrix):

    sentence_input = Input(shape=(length,),dtype='int32')
    
    word_embedding_layer = Embedding(word_embedding_matrix.shape[0], 300, weights=[word_embedding_matrix],trainable=True)
    word_vecs = word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Attention(20,20)([droped_vecs]*3)
    droped_rep = Dropout(0.2)(word_rep)
    title_vec = AttentivePooling(length,400)(droped_rep)

    sentEncodert = Model(sentence_input, title_vec)
    return sentEncodert

def get_entity_encoder(length,entity_dict):

    sentence_input = Input(shape=(length,),dtype='int32')
    
    word_embedding_layer = Embedding(len(entity_dict)+1, 300,trainable=True)
    word_vecs = word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Attention(5,40)([droped_vecs]*3)
    droped_rep = Dropout(0.2)(word_rep)   
    title_vec = AttentivePooling(length,200)(droped_rep)

    sentEncodert = Model(sentence_input, title_vec)
    return sentEncodert


def get_news_encoder(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict):

    news_input = Input(shape=(MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY,),dtype='int32')
    
    title_input = Lambda(lambda x:x[:,:MAX_TITLE])(news_input)
    vert_input = Lambda(lambda x:x[:,MAX_TITLE:MAX_TITLE+1])(news_input) 
    subvert_input = Lambda(lambda x:x[:,MAX_TITLE+1:MAX_TITLE+2])(news_input) 
    content_input = Lambda(lambda x:x[:,MAX_TITLE+2:MAX_TITLE+2+MAX_CONTENT])(news_input) 
    entity_input = Lambda(lambda x:x[:,MAX_TITLE+2+MAX_CONTENT:])(news_input) 

    title_encoder = get_text_encoder(MAX_TITLE,title_word_embedding_matrix)
    content_encoder = get_text_encoder(MAX_CONTENT,content_word_embedding_matrix)
    entity_encoder = get_entity_encoder(MAX_ENTITY,entity_dict)

    vert_embedding_layer = Embedding(len(category_dict)+1, 128,trainable=True)
    subvert_embedding_layer = Embedding(len(subcategory_dict)+1, 128,trainable=True)


    vert_vec = vert_embedding_layer(vert_input)
    subvert_vec = subvert_embedding_layer(subvert_input)
    vert_vec = Reshape((128,))(vert_vec)
    subvert_vec = Reshape((128,))(subvert_vec)
    vert_vec = Dense(128)(vert_vec)
    subvert_vec = Dense(128)(subvert_vec)
    vert_vec = Dropout(0.2)(vert_vec)
    subvert_vec = Dropout(0.2)(subvert_vec)

    
    title_vec = title_encoder(title_input)
    content_vec = content_encoder(content_input)
    entity_vec = entity_encoder(entity_input)

    vec = Concatenate(axis=-1)([title_vec,content_vec,vert_vec,subvert_vec,entity_vec])
    vec = Dense(400,activation='relu')(vec)

    sentEncodert = Model(news_input, vec)
    return sentEncodert



def news_level_pooling():
    vecs_input = Input(shape=(MAX_TITLE+2,120))
    
    title_input = Lambda(lambda x:x[:,:MAX_TITLE,:])(vecs_input) #(bz,50,30)
    vert_input = Lambda(lambda x:x[:,MAX_TITLE:MAX_TITLE+1,:])(vecs_input) #(bz,50,1,)
    subvert_input = Lambda(lambda x:x[:,MAX_TITLE+1:MAX_TITLE+2,:])(vecs_input) #(bz,50,1)
    
    
    title_vec = AttentivePooling(MAX_TITLE,120)(title_input)
    vert_vec = Reshape((120,))(vert_input)
    subvert_vec = Reshape((120,))(subvert_input)
    
    vec = Concatenate(axis=-1)([title_vec,vert_vec,subvert_vec])
    
    vec = Dense(40)(vec)
    
    return Model(vecs_input,vec)


def get_user_encoder():
    user_vecs_input = Input(shape=(MAX_CLICK,400))    
    
    user_vecs = Dropout(0.2)(user_vecs_input)

    user_vecs = Attention(20,20)([user_vecs]*3)
    user_vec = AttentivePooling(MAX_CLICK,400)(user_vecs)
#     user_vec = Dense(400,activation='relu')(user_vec)
    user_vec = Dense(370)(user_vec)
    return Model(user_vecs_input,user_vec)

def get_flaten_user_encoder(title_word_embedding_matrix,category_dict,subcategory_dict):
    user_vecs_input = Input(shape=(MAX_CLICK,MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY))    
     
    title_input = Lambda(lambda x:x[:,:,:MAX_TITLE])(user_vecs_input) #(bz,50,30)
    vert_input = Lambda(lambda x:x[:,:,MAX_TITLE:MAX_TITLE+1])(user_vecs_input) #(bz,50,1,)
    subvert_input = Lambda(lambda x:x[:,:,MAX_TITLE+1:MAX_TITLE+2])(user_vecs_input) #(bz,50,1)
        
    word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], 300, weights=[title_word_embedding_matrix],trainable=True)
    vert_embedding_layer = Embedding(len(category_dict)+1, 300,trainable=True)
    subvert_embedding_layer = Embedding(len(subcategory_dict)+1, 300,trainable=True)

    title_vecs = word_embedding_layer(title_input)
    vert_vecs = vert_embedding_layer(vert_input)
    subvert_vecs = subvert_embedding_layer(subvert_input)
    
    user_vecs = Concatenate(axis=-2)([title_vecs,vert_vecs,subvert_vecs])

    user_vecs = Reshape((MAX_CLICK*(MAX_TITLE+2),300))(user_vecs)

    user_vecs = Dropout(0.2)(user_vecs)

    user_vecs = Fastformer(3,40)([user_vecs]*3)
    
    user_vecs = Dropout(0.2)(user_vecs)
    
    user_vecs = Reshape((MAX_CLICK,MAX_TITLE+2,120))(user_vecs)
    
    user_vecs = TimeDistributed(news_level_pooling())(user_vecs) #(50,400)
        
    user_vecs = Dropout(0.2)(user_vecs)

    user_vec = AttentivePooling(MAX_CLICK,40)(user_vecs)
    user_vec = Dense(30)(user_vec)
    user_vec = Dropout(0.2)(user_vec)
    
    return Model(user_vecs_input,user_vec)

def create_model(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict):
        
    news_encoder = get_news_encoder(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict)
    user_encoder1 = get_user_encoder()
    flaten_user_encoder = get_flaten_user_encoder(title_word_embedding_matrix,category_dict,subcategory_dict)
    
    clicked_title_input = Input(shape=(MAX_CLICK,MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY,), dtype='int32')    
    title_inputs = Input(shape=(1+npratio,MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY,),dtype='int32') 

    user_vecs = TimeDistributed(news_encoder)(clicked_title_input)
    user_vec1 = user_encoder1(user_vecs)

    user_vec2 = flaten_user_encoder(clicked_title_input)
    
    user_vec = Concatenate(axis=-1)([user_vec1,user_vec2])
    
    news_vecs = TimeDistributed(news_encoder)(title_inputs)
    news_vecs = Dropout(0.2)(news_vecs)
    
    scores = keras.layers.Dot(axes=-1)([news_vecs,user_vec])
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     

    model = Model([title_inputs, clicked_title_input,],logits) 
    
    user_encoder = Model(clicked_title_input,user_vec)

    model = multi_gpu_model(model,gpus=2)
    model.compile(loss=['categorical_crossentropy'],
                    optimizer=Adam(lr=0.0001), 
                    #optimizer= SGD(lr=0.01),
                    metrics=['acc'])

    return model,news_encoder,user_encoder

def create_model_cg(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict):
        
    news_encoder = get_news_encoder(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict)

    clicked_title_input = Input(shape=(MAX_CLICK,MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY,), dtype='int32')    
    title_inputs = Input(shape=(1+npratio,MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY,),dtype='int32') 

    user_vecs = TimeDistributed(news_encoder)(clicked_title_input)
    
    user_vecs = Attention(20,20)([user_vecs]*3)
    user_vec = AttentivePooling(MAX_CLICK,400)(user_vecs)

    user_vec = Dense(400,activation='relu')(user_vec)
    
    news_vecs = TimeDistributed(news_encoder)(title_inputs)
    news_vecs = Dropout(0.2)(news_vecs)
    
    scores = keras.layers.Dot(axes=-1)([news_vecs,user_vec])
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     

    model = Model([title_inputs, clicked_title_input,],logits) 
    
    user_encoder = Model(clicked_title_input,user_vec)

    model = multi_gpu_model(model,gpus=2)
    model.compile(loss=['categorical_crossentropy'],
                    optimizer=Adam(lr=0.0001), 
                    #optimizer= SGD(lr=0.01),
                    metrics=['acc'])

    return model,news_encoder,user_encoder


def create_model_fg(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict):
        
    news_encoder = get_news_encoder(title_word_embedding_matrix,content_word_embedding_matrix,entity_dict,category_dict,subcategory_dict)

    clicked_title_input = Input(shape=(MAX_CLICK,MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY,), dtype='int32')    
    title_inputs = Input(shape=(1+npratio,MAX_TITLE+2+MAX_CONTENT+MAX_ENTITY,),dtype='int32') 
    
    news_vecs = TimeDistributed(news_encoder)(title_inputs)
    news_vecs = Dropout(0.2)(news_vecs)
    
    # User Modeling
         
    title_input = Lambda(lambda x:x[:,:,:MAX_TITLE])(clicked_title_input) #(bz,50,30)
    vert_input = Lambda(lambda x:x[:,:,MAX_TITLE:MAX_TITLE+1])(clicked_title_input) #(bz,50,1,)
    subvert_input = Lambda(lambda x:x[:,:,MAX_TITLE+1:MAX_TITLE+2])(clicked_title_input) #(bz,50,1)
        
    word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], 300, weights=[title_word_embedding_matrix],trainable=True)
    vert_embedding_layer = Embedding(len(category_dict)+1, 300,trainable=True)
    subvert_embedding_layer = Embedding(len(subcategory_dict)+1, 300,trainable=True)

    title_vecs = word_embedding_layer(title_input)
    vert_vecs = vert_embedding_layer(vert_input)
    subvert_vecs = subvert_embedding_layer(subvert_input)
    
    user_vecs = Concatenate(axis=-2)([title_vecs,vert_vecs,subvert_vecs])

    user_vecs = Reshape((MAX_CLICK*(MAX_TITLE+2),300))(user_vecs)
    user_vecs = Dropout(0.2)(user_vecs)
    user_vecs = Fastformer(10,40)([user_vecs]*3)
    user_vecs = Dropout(0.2)(user_vecs)
    user_vecs = Reshape((MAX_CLICK,MAX_TITLE+2,400))(user_vecs)
    user_vecs = TimeDistributed(news_level_pooling())(user_vecs) #(50,400)        
    user_vecs = Dropout(0.2)(user_vecs)
    user_vec = AttentivePooling(MAX_CLICK,400)(user_vecs)    
    
    scores = keras.layers.Dot(axes=-1)([news_vecs,user_vec])
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     

    model = Model([title_inputs, clicked_title_input,],logits) 
    
    user_encoder = Model(clicked_title_input,user_vec)

    model = multi_gpu_model(model,gpus=2)
    model.compile(loss=['categorical_crossentropy'],
                    optimizer=Adam(lr=0.0001), 
                    #optimizer= SGD(lr=0.01),
                    metrics=['acc'])

    return model,news_encoder,user_encoder