import  numpy as np
from read_h5 import get_train_cates
import pickle
import os
ORIGIN_EMOTION=['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence'
,'Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement'
,'Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity'
,'Suffering','Surprise','Sympathy','Yearning']
ALL_EMOTION=['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence'
,'Disapproval','Disconnection','Disquiet','Doubt','Confusion','Embarrassment','Engagement'
,'Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity'
,'Suffering','Surprise','Sympathy','Yearning']
PRE_EMBEDDING_FILE=R'../pre_train_model/glove.6B.100d.txt'
EMO_EMBEDDING_FILE=R'../mid_output/emo_embedding.pickle'
ALL_DATA_FILE=R'../mid_output/all_feature_label.h5'
def times_2_prob(matrix):
    out=np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        line_times=matrix[i][i]
        for j in range(matrix.shape[1]):
            out[i][j]=matrix[i][j]/line_times
    return out
def sym_uniform(martix):
    sum_list=[np.sum(line) for line in martix]
    D_sqrt_inv=[i**(-0.5) for i in sum_list]
    return D_sqrt_inv@martix@D_sqrt_inv
def word_vec_corelation_m(vecs):
    out=np.zeros((vecs.shape[0],vecs.shape[0]))
    for i in range(vecs.shape[0]):
        for j in range(vecs.shape[0]):
            out[i][j]=np.linalg.norm(vecs[i]-vecs[j],ord=2)
    out=1-out/out.max()
    return out
def get_adj_matrix(ratio_wvec=0.5):
    # train_labels=get_train_cates()
    # emo_dim=train_labels.shape[1]
    # count_m=np.zeros((emo_dim,emo_dim))
    # for line in train_labels:
    #     for i in range(emo_dim):
    #         for j in range(emo_dim):
    #             if line[i]==1 and line[j]==1:
    #                 count_m[i][j]+=1
    # prob_m=times_2_prob(count_m)
    # if not os.path.exists(EMO_EMBEDDING_FILE):
    emo_strs=ALL_EMOTION
    emo_strs=[i.lower() for i in emo_strs]
    if os.path.exists(PRE_EMBEDDING_FILE):
        word2vec_dict={}
        with open(PRE_EMBEDDING_FILE,'r') as f:
            line = f.readline()
            while line:
                line_sep=line.split(' ')
                if line_sep[0] in emo_strs:
                    word2vec_dict[line_sep[0]]=np.asarray(line_sep[1:]).astype(np.float) 
                line = f.readline()
            word2vec_dict['disquietment']=word2vec_dict['disquiet']
            del word2vec_dict['disquiet']
            word2vec_dict['doubt/confusion']=0.5*word2vec_dict['doubt']+0.5*word2vec_dict['confusion']
            del word2vec_dict['doubt']
            del word2vec_dict['confusion']
            with open(EMO_EMBEDDING_FILE,'wb') as f_emo:
                pickle.dump(word2vec_dict,f_emo)
    else :
        with open(EMO_EMBEDDING_FILE,'rb') as f_emo:
            word2vec_dict=pickle.load(f_emo)
    wordvec_m=np.stack([word2vec_dict[emo.lower()]  for emo in ORIGIN_EMOTION])
    A_word=word_vec_corelation_m(wordvec_m)

if __name__=='__main__':
    get_adj_matrix()