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
        def __init__(self,h5_group_dict,word_vec_m,input_names,label_name,batch_size,shuffle=False):
            self.word_vec_m=word_vec_m
            self.h5_group=h5_group_dict
            self.inputs_names=input_names
            self.batch_size = batch_size
            self.features_dict={i:h5_group_dict[i] for i in input_names}
            self.labels=h5_group_dict[label_name]
            self.shuffle=shuffle
            self.on_epoch_end()
        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(self.labels.shape[0])
            if self.shuffle == True:
                print('shuffle')
                np.random.shuffle(self.indexes)
        def __len__(self):
            self.on_epoch_end()
            return math.ceil(self.labels.shape[0] / self.batch_size)

        def __getitem__(self, idx):
            ind=self.indexes[idx*self.batch_size:(idx + 1)*self.batch_size]
            batch_x = {k:np.asarray([v[i] for i in ind]) for k,v in self.features_dict.items()}
            batch_x['word_vector']=np.stack([self.word_vec_m]*list(batch_x.values())[0].shape[0])
            batch_y =np.asarray([self.labels[i] for i in ind])
            return batch_x,batch_y

if __name__=='__main__':
    count={i:0 for i in range(2048)}
    with tables.open_file(OUT_PATH,mode='r') as f:
        print(f.root.train.full_features[0])
        print(f.root.test.full_features[0])

        # for i in range(100):
        #     print(np.count_nonzero(np.reshape(f.root.train.full_features[i],(-1)),axis=-1).tolist()/np.reshape(f.root.train.full_features[i],(-1)).shape[0])
        # for i in range(100):
        #     arr=np.reshape(f.root.train.full_features[i],(-1,2048))
        #     for feature in arr:
        #         for index in range(2048):
        #             if feature[index]==0:
        #                 count[index]+=1
    #print(list(count.items()))
    #print(sorted(list(count.items()),key=lambda x:-x[1])) 
        # print(f.root.validation.full_features[0])

    