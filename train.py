from whole_model import whole_model
from read_h5 import data_generator,INPUT_NAMES,LABEL_NAME
from gcn_model import get_wordvec_m
import tensorflow as tf
import pickle
import tables
FEATURE_PATH=R'../mid_output/all_feature_label.h5'
tick='1000'
MODEL_PATH=R'../model/model_'+tick+'.h5'
HISTORY_PATH=R'../model/history_'+tick+'+.pickle'
def train():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
        except RuntimeError as e:
            print(e)
    model=whole_model()
    initializer=tf.keras.initializers.GlorotNormal()
    for l in model.layers:
        out=[]
        for i in l.get_weights():
            out.append(initializer(i.shape))
        l.set_weights(out)
    word_vec_m=get_wordvec_m()
    f=tables.open_file(FEATURE_PATH,mode='r')
    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy())
    history=model.fit(
        x=data_generator(f.root.train._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,32),
        epochs=20,
        verbose=2,
        validation_data=data_generator(f.root.validation._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,32)
    )
    model.save(MODEL_PATH)
    f.close()
    print(history.history.items())
    with open(HISTORY_PATH,'wb') as f_h:
        pickle.dump(history.history,f_h)
if __name__=='__main__':
    train()
    # print([i.shape for i in whole_model().inputs])
    # model=whole_model()
    # f=tables.open_file(FEATURE_PATH,mode='r')
    # word_vec_m=get_wordvec_m()
    # dg=data_generator(f.root.train._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,1)
    # print(len(dg))
    # model.predict(a)
    # f.close()
    # print([i.shape for i in a.values()])
    # with open(HISTORY_PATH,'rb') as f:
    #     his=pickle.load(f)
    # tep=zip(his['loss'],his['val_loss'])
    # for a,b in tep:
    #     print(a,b)
    
   