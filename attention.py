import tensorflow as tf
import tensorflow.keras.layers as layers 
import numpy as np
from math import ceil
class MatMul(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super().__init__(**kwargs)

  def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel',
                                  shape=(self.output_dim,input_shape[1]),
                                  initializer='uniform',
                                  trainable=True,
                                  dtype='float32')
    super().build(input_shape)
    
  def call(self, inputs):
    KT = tf.transpose(self.kernel) 
    IT = tf.transpose(inputs,perm=[0,2,1])
    NTMT = tf.reshape(tf.reshape(IT, [-1, IT.shape[-1]]) @ KT, [-1, IT.shape[-2], KT.shape[-1]])
    return  tf.transpose(NTMT,perm=[0,2,1])
  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'output_dim': self.output_dim,
      })
      return config
@tf.function
def calculate_distance(boundingbox,n_rows=7,n_cols=7,h=224,w=224,dis_type='manhattan'):
  x1,y1,x2,y2=boundingbox[0],boundingbox[1],boundingbox[2],boundingbox[3]
  y1,y2=y1/h*n_rows,y2/h*n_rows
  x1,x2=x1/w*n_cols,x2/w*n_cols
  y_range,x_range=tf.cast(tf.range(n_rows),tf.float32),tf.cast(tf.range(n_cols),tf.float32)
  zero_tensor=tf.constant(0.)
  out=tf.TensorArray(tf.float32, size=n_rows*n_cols,dynamic_size=False, clear_after_read=False)
  index=0
  for y in y_range:
      for x in x_range:
          d_y,d_x=tf.keras.backend.max(tf.stack((y-y1,y+1-y2,zero_tensor))),tf.keras.backend.max(tf.stack((x-x1,x+1-x2,zero_tensor)))
          d_y,d_x=tf.math.ceil(d_y),tf.math.ceil(d_x)
          if dis_type=='manhattan':
            out.write(index,tf.reshape(d_y+d_x,[]))
          else:
            out.write(index,tf.reshape(d_y+d_x,[]))
          index+=1
  return out.stack()

def attention_fusion(alpha=0.5,n_rows=7,n_cols=7,v_size=2048):
  transpose_layer=layers.Lambda(lambda x: tf.transpose(x,perm=[0,2,1]),name='transpose')
  full_features_input=tf.keras.Input(shape=(n_rows,n_cols,v_size), name="full_features",dtype='float32')
  full_features=layers.Reshape((n_rows*n_cols,v_size),name='flattern1')(full_features_input)
  full_features=transpose_layer(full_features)
  V_e=MatMul(n_rows*n_cols)(full_features)
  Q_e=MatMul(n_rows*n_cols)(full_features)
  M_e=layers.Dot(axes=(1,1))([V_e,Q_e]) 
  alpha_e=layers.Softmax(axis=-1)(M_e)
  f_e=layers.Dot(axes=(2,1))([full_features,alpha_e])
  self_attention_output=layers.GlobalAveragePooling1D(data_format='channels_first')(f_e)
  self_attention_model=tf.keras.Model(inputs=full_features_input,outputs=self_attention_output)

  env_features_input=tf.keras.Input(shape=(n_rows,n_cols,v_size), name="env_features")
  env_features=layers.Reshape((n_rows*n_cols,v_size),name='flattern2')(env_features_input)
  person_features_input=tf.keras.Input(shape=(v_size), name="person_features")
  person_product=layers.Dense(n_rows*n_cols)(person_features_input)
  person_product=layers.Reshape((1,n_rows*n_cols))(person_product)
  person_product=tf.repeat(person_product,[n_rows*n_cols],axis=-2,name='expand')
  person_product=transpose_layer(person_product)
  bbox_input=tf.keras.Input(shape=(4), name="body_bboxs")
  d=layers.Lambda(lambda bb:tf.map_fn(calculate_distance,bb))(bbox_input)
  d=tf.expand_dims(d,-1)
  I_d=layers.Concatenate(axis=-1)([env_features,d])
  I_d=transpose_layer(I_d)
  I_d_product=MatMul(n_rows*n_cols)(I_d)
  product_sum=layers.Add()([person_product,I_d_product])
  M_c=tf.keras.activations.tanh(product_sum)
  W_cM_c=MatMul(1)(M_c)  
  alpha_c=layers.Softmax(axis=-1)(W_cM_c)
  f_c=layers.Dot(axes=(2,1))([alpha_c,env_features])
  f_c=layers.Reshape((v_size,))(f_c)
  position_aware_attetion_model=tf.keras.Model(
    inputs=[env_features_input,person_features_input,bbox_input],outputs=f_c)
  fusion_output=alpha*self_attention_output+(1-alpha)*f_c
  attention_fusion_model=tf.keras.Model(
    inputs=[full_features_input,env_features_input,person_features_input,bbox_input],outputs=fusion_output)
  return attention_fusion_model
    
if __name__=='__main__':
  model=attention_fusion()
  model.summary()
  # l=MatMul(4)
  # l.build([None,3,2])
  # l.set_weights([np.asarray([[1,0,0],[1,0,0],[1,0,0],[1,0,0]])])
  # x=np.random.rand(2,3,2)
  # print(x)
  # print(l(x))