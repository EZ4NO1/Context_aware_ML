import numpy as np
import tensorflow as tf
from attention import MatMul,attention_fusion
from gcn_model import get_wordvec_m,tow_layers_GNN_model,get_adj_matrix
MODEL_FIG=R'../model/model_fig.png'
def whole_model():
    wordvec_m=get_wordvec_m()
    attention_fusion_model=attention_fusion()
    fusion_output=attention_fusion_model.outputs[0]
    fusion_output=tf.expand_dims(fusion_output,-1)
    gcn_model=tow_layers_GNN_model(get_adj_matrix())
    gcn_output=gcn_model.output
    mul_output=tf.keras.layers.Dot(axes=[2,1],name='Matmul_attention_gcn')([gcn_output,fusion_output])
    mul_output=tf.keras.layers.Reshape((wordvec_m.shape[0],))(mul_output)
    person_feature=attention_fusion_model.get_layer('person_features').output
    cat_feature=tf.keras.layers.Concatenate()([mul_output,person_feature])
    D1_output=tf.keras.layers.Dense(1000,activation='tanh')(cat_feature)
    out=tf.keras.layers.Dense(wordvec_m.shape[0],activation='sigmoid')(D1_output)
    model=tf.keras.Model(inputs=attention_fusion_model.inputs+gcn_model.inputs,outputs=out)
    return model
if __name__ == "__main__":
    model=whole_model()
    model.summary()
    #print(model.inputs)
    #tf.keras.utils.plot_model(model,to_file=MODEL_FIG)  