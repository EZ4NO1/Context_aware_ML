import tensorflow as tf
import numpy as np
from read_h5 import data_generator,INPUT_NAMES,LABEL_NAME
from attention import MatMul
from gcn_model import GCN_Layer
from gcn_model import get_wordvec_m
import tables
import os
import matplotlib.pyplot as plt
MODEL_PATH=R'../model/mode.h5.200_back'
DATA_PATH=R'../mid_output/all_feature_label.h5'
PRE_LABEL_PATH=os.path.join(R'../output',MODEL_PATH[MODEL_PATH.rfind('.')+1:]+'.npy')
MAP_IMG=os.path.join(R'../output',MODEL_PATH[MODEL_PATH.rfind('.')+1:]+'.png')
def mAP(pre_labels,groud_true,split_num=20):
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


if __name__ == "__main__":
    # print(PRE_LABEL_PATH)
    if not os.path.exists(PRE_LABEL_PATH):
        model=tf.keras.models.load_model(MODEL_PATH,custom_objects={'MatMul':MatMul,'GCN_Layer':GCN_Layer})
        f=tables.open_file(DATA_PATH,mode='r')
        word_vec_m=get_wordvec_m()
        y=model.predict(
            x=data_generator(f.root.test._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,32),
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
    map_out=mAP(y,ground_true)
    save_map_img(MAP_IMG,map_out)