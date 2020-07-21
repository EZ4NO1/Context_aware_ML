import  numpy as np
from read_h5 import get_train_cates
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.keras.layers as layers 

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
COND_PROB_FILE=R'../mid_output/conditional_prob_m.npy'
CORELATION_HEAT_MAP=R'../mid_output/corelation_heat_map.jpg'
def times_2_prob(matrix):
    out=np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        line_times=matrix[i][i]
        for j in range(matrix.shape[1]):
            out[i][j]=matrix[i][j]/line_times
    return out
def sym_uniform(martix):
    sum_list=[np.sum(line) for line in martix]
    D_list=[i**(-0.5) for i in sum_list]
    D_sqrt_inv=np.zeros(martix.shape)
    np.fill_diagonal(D_sqrt_inv,D_list)
    return D_sqrt_inv@martix@D_sqrt_inv
def word_vec_corelation_m(vecs):
    out=np.zeros((vecs.shape[0],vecs.shape[0]))
    for i in range(vecs.shape[0]):
        for j in range(vecs.shape[0]):
            out[i][j]=np.linalg.norm(vecs[i]-vecs[j],ord=2)
    out=1-out/out.max()
    return out
def get_wordvec_m():
    emo_strs=ALL_EMOTION
    emo_strs=[i.lower() for i in emo_strs]
    if not os.path.exists(PRE_EMBEDDING_FILE):
        word2vec_dict={}
        with open(PRE_EMBEDDING_FILE,'r') as f:
            line = f.readline()
            while line:
                line_sep=line.split(' ')
                if line_sep[0] in emo_strs:
                    word2vec_dict[line_sep[0]]=np.asarray(line_sep[1:]).astype(np.float32) 
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
    return wordvec_m
def get_adj_matrix(ratio_wvec=0.5,diag_ratio=0.5):
    wordvec_m=get_wordvec_m()
    A_word=word_vec_corelation_m(wordvec_m)

    if not os.path.exists(COND_PROB_FILE):
        train_labels=get_train_cates()
        emo_dim=train_labels.shape[1]
        count_m=np.zeros((emo_dim,emo_dim))
        for line in train_labels:
            for i in range(emo_dim):
                for j in range(emo_dim):
                    if line[i]==1 and line[j]==1:
                        count_m[i][j]+=1
        prob_m=times_2_prob(count_m)
        np.save(COND_PROB_FILE,prob_m)
    else:
        prob_m=np.load(COND_PROB_FILE)
    A_prob=prob_m
    A=ratio_wvec*A_word+(1-ratio_wvec)*A_prob
    for i in range(A.shape[0]):
        A[i][i]=A[i][i]*diag_ratio
    A_hat=sym_uniform(A)
    return A_hat.astype(np.float32)
def draw_corelation_heat_map(matrix,ticks):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix,cmap='gray')
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_yticks(np.arange(len(ticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(ticks)):
    #     for j in range(len(ticks)):
    #         text = ax.text(i,j,matrix[i][j],
    #                     ha="center", va="center", color="w")
    ax.set_title("Multi-Label Corelation")
    # # ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis   
    # ax.yaxis.tick_left()    
    fig.tight_layout()
    plt.savefig(CORELATION_HEAT_MAP,quality=100,dpi=fig.dpi)


class GCN_Layer(layers.Layer):
    def __init__(self, output_dim,Adj,activation='relu',**kwargs):
        self.output_dim = output_dim
        self.Adj=tf.constant(Adj)
        self.activation=activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[-1],self.output_dim),
                                  initializer='uniform',
                                  trainable=True,
                                  dtype='float32')
        super().build(input_shape)
    def call(self, inputs):
        Adj_T=self.Adj   #A_T*H
        input_T=tf.transpose(inputs,perm=[0,2,1])
        M_T=tf.reshape(tf.reshape(input_T, [-1, input_T.shape[-1]]) @Adj_T, [-1, input_T.shape[-2], Adj_T.shape[-1]])
        M=tf.transpose(M_T,perm=[0,2,1])
        OUT=tf.reshape(tf.reshape(M,[-1,M.shape[-1]])@self.kernel,[-1,M.shape[-2],self.kernel.shape[-1]])
        return  layers.Activation(self.activation)(OUT)
    def get_config(self):
      config = super().get_config().copy()
      config.update({
          'output_dim': self.output_dim,
          'Adj':self.Adj.numpy(),
          'activation':self.activation
      })
      return config
def tow_layers_GNN_model(Adj,node_feature_dim=2048):
    wordvec_s=get_wordvec_m().shape
    input_v=layers.Input(shape=wordvec_s, name="word_vector",dtype='float32')
    M1=GCN_Layer(wordvec_s[-1],Adj,'tanh')(input_v)
    M2=GCN_Layer(node_feature_dim,Adj,'tanh')(M1)
    model=tf.keras.Model(inputs=input_v,outputs=M2)
    return model
if __name__=='__main__':
    A_hat=get_adj_matrix()
    #print(A_hat)
    # draw_corelation_heat_map(A_hat[:10,:10],ORIGIN_EMOTION[:10])
    draw_corelation_heat_map(A_hat,ORIGIN_EMOTION)
    model=tow_layers_GNN_model(A_hat)
    model.summary()