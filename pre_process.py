from scipy.io import loadmat
import os.path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tables
from tqdm import tqdm
STRUCT_FILE=R'../Data_set/annotations/Annotations.mat'
IMG_PATH=R'../Data_set/emotic'
OUT_PATH=R'../mid_output/all_feature_label.h5'
IMAGENET_PATH=R'../pre_train_model/resnet101v2_weights_tf_dim_ordering_tf_kernels.h5'
BATCH_SIZE=200
ALL_EMOTION=['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence'
,'Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement'
,'Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity'
,'Suffering','Surprise','Sympathy','Yearning']
def show_img_with_box(img,body_bbox=None,index=1):
    ax = plt.subplot(2,2,index)
    plt.imshow(img.numpy())
    if body_bbox is not None:
        rec=patches.Rectangle((body_bbox[0],body_bbox[1]),body_bbox[2]-body_bbox[0],body_bbox[3]-body_bbox[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rec)
def as_ndarray(x):
    return x if isinstance(x,np.ndarray) else np.asarray([x])
def preprocess():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
        except RuntimeError as e:
            print(e)
    ResNet101=tf.keras.applications.ResNet101V2(include_top=True, weights=IMAGENET_PATH)
    layer_in=ResNet101.layers[0].input
    layer_out=ResNet101.layers[-3].output
    ResNet101=tf.keras.Model(inputs=layer_in,outputs=layer_out)
    # ResNet101.summary()

    CATE_DICT=dict(enumerate(ALL_EMOTION))
    I_CATE_DICT={v:k for k,v in CATE_DICT.items()}
    mat_struct=loadmat(STRUCT_FILE,struct_as_record=False,squeeze_me=True)
    train_struct,val_struct,test_struct=mat_struct['train'],mat_struct['val'],mat_struct['test']
    target_img_size=224
    skip_num=0
    h5file=tables.open_file(OUT_PATH,mode='w')
    h5_groups=[h5file.create_group("/",'train'),h5file.create_group("/",'validation'),h5file.create_group("/",'test')]
    for struct_whole,h5_group in tqdm(zip([train_struct,val_struct,test_struct],h5_groups),ncols=100):
        earray_full=h5file.create_earray(h5_group,'full_features',atom=tables.FloatAtom(),shape=(0,7,7,2048))
        earray_person=h5file.create_earray(h5_group,'person_features',atom=tables.FloatAtom(),shape=(0,2048))
        earray_env=h5file.create_earray(h5_group,'env_features',atom=tables.FloatAtom(),shape=(0,7,7,2048))
        earray_combined_cates=h5file.create_earray(h5_group,'combined_cates',atom=tables.IntAtom(),shape=(0,len(CATE_DICT.items())))
        earray_ave_cates=h5file.create_earray(h5_group,'ave_cates',atom=tables.FloatAtom(),shape=(0,len(CATE_DICT.items())))
        earray_body_bboxs=h5file.create_earray(h5_group,'body_bboxs',atom=tables.IntAtom(),shape=(0,4))
        for struct in tqdm(np.array_split(struct_whole,struct_whole.size//BATCH_SIZE),ncols=100):
            full_imgs=[]
            person_imgs=[]
            env_imgs=[]
            combined_cates=[]
            ave_cates=[]
            body_bboxs=[]
            for i in tqdm(struct,ncols=100):
                filename,folder,n_col,n_row,persons=i.filename,i.folder,i.image_size.n_col,i.image_size.n_row,i.person
                full_file_name=os.path.join(IMG_PATH,folder,filename)
                persons=as_ndarray(persons)
                for person in persons:
                    try:
                        full_image=tf.io.decode_jpeg(tf.io.read_file(full_file_name),channels=3)/255
                    except Exception as e:
                        skip_num+=1
                        print(e)
                        continue
                    body_bbox=person.body_bbox.astype(int)
                    body_bbox[0],body_bbox[1],body_bbox[2],body_bbox[3]=max(0,body_bbox[0]),max(0,body_bbox[1]),min(body_bbox[2],n_col-1),min(body_bbox[3],n_row-1)
                    annotations=as_ndarray(person.annotations_categories)
                    categories=[as_ndarray(ann.categories) for ann in annotations]
                    sum_cate=np.zeros(len(CATE_DICT.items()))
                    for cate in categories:
                        for emo in cate:
                            sum_cate[I_CATE_DICT[emo]]+=1
                    combined_cate=np.asarray([0 if j==0 else 1 for j in sum_cate])
                    ave_cate=sum_cate/len(categories)
                    combined_cates.append(combined_cate)
                    ave_cates.append(ave_cate)

                    person_image=tf.image.crop_to_bounding_box(full_image,body_bbox[1],body_bbox[0],body_bbox[3]-body_bbox[1],body_bbox[2]-body_bbox[0])
                    env_image=full_image.numpy()
                    env_image[body_bbox[1]:body_bbox[3]+1,body_bbox[0]:body_bbox[2]+1,:]=1
                    env_image=tf.constant(env_image)

                    full_image,person_image,env_image=tf.image.resize(full_image,[target_img_size,target_img_size])\
                        ,tf.image.resize(person_image,[target_img_size,target_img_size]),tf.image.resize(env_image,[target_img_size,target_img_size])
                    body_bbox[0],body_bbox[2]=body_bbox[0]/n_col*target_img_size,body_bbox[2]/n_col*target_img_size
                    body_bbox[1],body_bbox[3]=body_bbox[1]/n_col*target_img_size,body_bbox[3]/n_col*target_img_size
                    body_bbox=body_bbox.astype(int)
                    full_imgs.append(full_image)
                    person_imgs.append(person_image)
                    env_imgs.append(env_image)
                    body_bboxs.append(body_bbox)
            full_features=ResNet101.predict(np.stack(full_imgs))
            person_features=ResNet101.predict(np.stack(person_imgs))
            person_features=tf.math.reduce_mean(person_features,axis=[1,2]).numpy()
            env_features=ResNet101.predict(np.stack(env_imgs))
            combined_cates=np.stack(combined_cates)
            ave_cates=np.stack(ave_cates)
            body_bboxs=np.stack(body_bboxs)
            #print(full_features.shape)
            earray_full.append(full_features)
            earray_person.append(person_features)
            earray_env.append(env_features)
            earray_combined_cates.append(combined_cates)
            earray_ave_cates.append(ave_cates)
            earray_body_bboxs.append(body_bboxs)
    print('skip_instance_num',skip_num)
    h5file.close()
if __name__=='__main__':
    preprocess()


    