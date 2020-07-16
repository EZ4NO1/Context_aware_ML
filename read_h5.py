import tables
OUT_PATH=R'../mid_output/all_feature_label.h5'

def get_train_cates():
    with tables.open_file(OUT_PATH,mode='r') as f:
        A=f.root.train.combined_cates
        A_numpy=A[:]
        return A_numpy
if __name__=='__main__':
    with tables.open_file(OUT_PATH,mode='r') as f: 
        print(f.root.train._v_children.keys())
        A=f.root.train.combined_cates
        A_numpy=A[:]
        print(A_numpy.shape)
    