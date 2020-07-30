import tables
import tensorflow as tf
import math
import numpy as np
OUT_PATH=R'../mid_output/all_feature_label.h5'
INPUT_NAMES=['full_features','person_features','env_features','body_bboxs']
LABEL_NAME='combined_cates'
def get_train_cates():
    with tables.open_file(OUT_PATH,mode='r') as f:
        A=f.root.train.combined_cates
        A_numpy=A[:]
        return A_numpy
def get_train_all_inputs():
    with tables.open_file(OUT_PATH,mode='r') as f:
        dic=f.root.train._v_children
        del dic['combined_cates']
        del dic['ave_cates']
        #dic={k:v[:] for k,v in dic.items()}
        t=dic['full_features'][3:9]
        print(dic.keys())
class data_generator(tf.keras.utils.Sequence):
        def __init__(self,h5_group_dict,word_vec_m,input_names,label_name,batch_size):
            self.word_vec_m=word_vec_m
            self.h5_group=h5_group_dict
            self.inputs_names=input_names
            self.batch_size = batch_size
            self.features_dict={i:h5_group_dict[i] for i in input_names}
            self.labels=h5_group_dict[label_name]
        def __len__(self):
            return math.ceil(self.labels.shape[0] / self.batch_size)

        def __getitem__(self, idx):
            batch_x = {k:v[idx*self.batch_size:(idx + 1)*self.batch_size] for k,v in self.features_dict.items()}
            batch_x['word_vector']=np.stack([self.word_vec_m]*list(batch_x.values())[0].shape[0])
            batch_y = self.labels[idx*self.batch_size:(idx + 1)*self.batch_size]
            return batch_x,batch_y

if __name__=='__main__':
    with tables.open_file(OUT_PATH,mode='r') as f:
        print(f.root.train.body_bboxs[3])
    