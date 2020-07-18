import numpy as np
import tensorflow as tf
from attention import MatMul,attention_fusion
from gcn_model import get_wordvec_m,tow_layers_GNN_model,get_adj_matrix
def whole_model():
    wordvec_m=get_wordvec_m()
    attention_fusion_model=attention_fusion()
    fusion_output=attention_fusion_model.outputs[0]
    gcn_model=tow_layers_GNN_model(get_adj_matrix())
    gcn_output=gcn_model.outputs[0]
    mul_output=tf.reshape(gcn_output@tf.expand_dims(fusion_output,-1),[-1,gcn_output.shape[-2]])
    person_feature=attention_fusion_model.get_layer('person_features').output
    cat_feature=tf.keras.layers.Concatenate()([mul_output,person_feature])
    D1_output=tf.keras.layers.Dense(wordvec_m.shape[0],activation='sigmoid')(cat_feature)
    out=tf.keras.layers.Dense(wordvec_m.shape[0],activation='sigmoid')(D1_output)
    model=tf.keras.Model(inputs=attention_fusion_model.inputs+gcn_model.inputs,outputs=out)
    return model
if __name__ == "__main__":
    model=whole_model()
    model.summary()
    print(model.inputs)  