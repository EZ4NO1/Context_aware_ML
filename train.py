from whole_model import whole_model
from read_h5 import data_generator,INPUT_NAMES,LABEL_NAME
from gcn_model import get_wordvec_m
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow as tf
import pickle
import tables
from tensorflow.keras.callbacks import LambdaCallback

FEATURE_PATH=R'../mid_output/all_feature_label.h5'
tick='1000'
MODEL_PATH=R'../model/model_'+tick+'.h5'
HISTORY_PATH=R'../model/history_'+tick+'+.pickle'
def loss_fn_with_l2(input_tensor):
    def custom_loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) + K.mean(input_tensor)
    return custom_loss
def train(with_reg=False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        except RuntimeError as e:
            print(e)
    model=whole_model()
    # initializer=tf.keras.initializers.GlorotNormal()
    # for l in model.layers:
    #     out=[]
    #     for i in l.get_weights():
    #         print(i)
    #         out.append(initializer(i.shape))
    #     l.set_weights(out)
    word_vec_m=get_wordvec_m()
    f=tables.open_file(FEATURE_PATH,mode='r')
    if with_reg:
        def add_l2_regularization(layer):
            def _add_l2_regularization():
                l2 = tf.keras.regularizers.l2(1e-4)
                return l2(layer.kernel)
            return _add_l2_regularization

        for l in model.layers:
            if hasattr(l,'kernel'):
                model.add_loss(add_l2_regularization(l))
        print(len(model.losses))
        model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=tf.keras.metrics.BinaryCrossentropy())
    else:
        model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy())
    #print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.losses))
    #print(len(model.losses))
    history=model.fit(
        x=data_generator(f.root.train._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,32),
        epochs=50,
        verbose=2,
        validation_data=data_generator(f.root.validation._v_children,word_vec_m,INPUT_NAMES,LABEL_NAME,32),
        #callbacks = [print_weights]
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
    # with open(HISTORY_PATH,'rb') as f_h:
    #     tep=pickle.load(f_h)
    # for i,j in zip(tep['loss'],tep['val_loss']):
    #     print(i,j)