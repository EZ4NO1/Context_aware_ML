import tensorflow as tf
import numpy as np
from read_h5 import data_generator,INPUT_NAMES,LABEL_NAME
from attention import MatMul
from gcn_model import GCN_Layer
from gcn_model import get_wordvec_m
import tables
import os
import matplotlib.pyplot as plt
import pickle
MODEL_PATH=R'../model/model_1000.h5'
DATA_PATH=R'../mid_output/all_feature_label.h5'
PRE_LABEL_PATH=os.path.join(R'../output',MODEL_PATH[MODEL_PATH.rfind('.')+1:]+'.npy')
MAP_IMG=os.path.join(R'../output',MODEL_PATH[MODEL_PATH.rfind('.')+1:]+'.png')
ALL_EMOTION=['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence'
,'Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement'
,'Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity'
,'Suffering','Surprise','Sympathy','Yearning']
def mAP_old(pre_labels,groud_true,split_num=20):
    label_num=pre_labels.shape[-1]
    instance_num=pre_labels.shape[0]
    out=[]
    for i in range(split_num):
        TP,TN,FP,FN=([0]*label_num for i in range(4))
        delimeter=1/split_num*i
        for j in range(instance_num):
            for k in range(label_num):
                if groud_true[j][k]==1:
                    if pre_labels[j][k]>delimeter:
                        TP[k]+=1
                    else:
                        FN[k]+=1
                else:
                    if pre_labels[j][k]>delimeter:
                        FP[k]+=1
                    else:
                        TN[k]+=1
        # print(TP,FP,FN,sep='\n')
        # print('\n')
        P=[i/(i+j) for i,j in zip(TP,FP) if i+j!=0 ]
        R=[i/(i+j)  for i,j in zip(TP,FN) if i+j!=0]
        P=sum(P)/len(P)
        R=sum(R)/len(R)
        out.append((P,R))
    return out
def save_map_img(filename,mpa_result):
    x=[i for i,_ in mpa_result]
    y=[i for _,i in mpa_result]
    plt.plot(x,y)
    plt.savefig(filename)

def AveragePrecision(pre_labels,ground_true):
    label_num=pre_labels.shape[-1]
    output=[]
    for i in range(label_num):
        output.append(AveragePrecision_(pre_labels[:,i],ground_true[:,i]))
    return output

def AveragePrecision_(pre_labels,ground_true):
    indices=np.argsort(pre_labels)[::-1]
    #print('indices',indices.shape)
    #for i in indices:
    #    print(ground_true[i])
    #return
    pos_count = 0
    total_count = 0
    precision_at_i = 0.

    for i in indices:
        total_count += 1
        label = ground_true[i]
        if label == 0:
            continue
        if label == 1:
            pos_count += 1
        if label == 1:
            precision_at_i += float(pos_count) / total_count
    precision_at_i /= pos_count
    #print(pos_count,total_count)
    #print(precision_at_i)
    return precision_at_i

def evaluate(model):
    word_vec_m=get_wordvec_m()
    y=model.predict(
            x=data_generator(f.root.test._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,32,shuffle=False),
            verbose=1
        )
    with tables.open_file(DATA_PATH,mode='r') as f:
        ground_true=f.root.test.combined_cates[:]
    map_out=AveragePrecision(y,ground_true)
    return sum(map_out)/len(map_out)
    

if __name__ == "__main__":
    # print(PRE_LABEL_PATH)
    # with open(R'../model/history.pickle.200_back','rb') as f_h:
    #     tep=pickle.load(f_h)
    # for i,j in zip(tep['loss'],tep['val_loss']):
    #     print(i,j)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)
    if not os.path.exists(PRE_LABEL_PATH):
        model=tf.keras.models.load_model(MODEL_PATH,custom_objects={'MatMul':MatMul,'GCN_Layer':GCN_Layer})
        f=tables.open_file(DATA_PATH,mode='r')
        word_vec_m=get_wordvec_m()
        y=model.predict(
            x=data_generator(f.root.test._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,32,shuffle=False),
            verbose=1
        )
        f.close()
        np.save(PRE_LABEL_PATH,y)
    else:
        y=np.load(PRE_LABEL_PATH)
    with tables.open_file(DATA_PATH,mode='r') as f:
        ground_true=f.root.test.combined_cates[:]
    bce = tf.keras.losses.BinaryCrossentropy()
    print('binary cross entropy:\n   ',bce(ground_true, y).numpy())
    map_out=AveragePrecision(y,ground_true)
    for i,j in zip(ALL_EMOTION,map_out):
        print(i,'   ',j)
    print(sum(map_out)/len(map_out))
    # save_map_img(MAP_IMG,map_out)
    #AveragePrecision_(y[:,1],ground_true[:,1])
    #print(ground_true.shape)