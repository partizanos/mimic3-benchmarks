import os 
import pandas as pd 
import torch
import pickle 
import numpy as np 


def read_pickle_file(pickle_file):
    """ Read pickle file.
    """
    print(pickle_file)
    with open(pickle_file, "rb") as f:
        fcontent =  pickle.load(f)
        f.close()
        return fcontent
    

def read_statistical_features(FOLDER_PATH):
    data = {}
    data['test_names'] = read_pickle_file(os.path.join(FOLDER_PATH,'test_names'))
    data['test_ts'] = read_pickle_file(os.path.join(FOLDER_PATH,'test_ts'))
    data['test_X'] = read_pickle_file(os.path.join(FOLDER_PATH,'test_X'))
    data['test_y'] = read_pickle_file(os.path.join(FOLDER_PATH,'test_y'))
    data['train_name'] = read_pickle_file(os.path.join(FOLDER_PATH,'train_name'))
    data['train_ts'] = read_pickle_file(os.path.join(FOLDER_PATH,'train_ts'))
    data['train_X'] = read_pickle_file(os.path.join(FOLDER_PATH,'train_X'))
    data['train_y'] = read_pickle_file(os.path.join(FOLDER_PATH,'train_y'))
    data['val_names'] = read_pickle_file(os.path.join(FOLDER_PATH,'val_names'))
    data['val_ts'] = read_pickle_file(os.path.join(FOLDER_PATH,'val_ts'))
    data['val_X'] = read_pickle_file(os.path.join(FOLDER_PATH,'val_X'))
    data['val_y'] = read_pickle_file(os.path.join(FOLDER_PATH,'val_y'))
    
    train_df = pd.DataFrame({
        'stat_features':data['train_X'].tolist(),
        'name':data['train_name'],
        # 'ts':data['train_ts'],
        'y':data['train_y'].tolist(),
        }
    )
    
    train_df.set_index('name',inplace=True)
    
    val_df = pd.DataFrame({
        'stat_features':data['val_X'].tolist(),
        'name':data['val_names'],
        # 'ts':data['val_ts'],
        'y':data['val_y'].tolist(),
        }
    )
    val_df.set_index('name',inplace=True)
    test_df = pd.DataFrame({
        'stat_features':data['test_X'].tolist(),
        'name':data['test_names'],
        # 'ts':data['test_ts'],
        'y':data['test_y'].tolist(),
        }
    )
    test_df.set_index('name',inplace=True)

    return train_df, val_df, test_df


def create_df(name,type,data):
        df= pd.DataFrame({
            'name':data[f'{name}_names'],
            f'{type}_embedding':data[f'{name}_embeddings'].tolist()
        })
        df.set_index('name', inplace=True)
        return df


def read_lstm_embeddings(folder):
    print('Reading lstm embeddings from folder')
    file_list = [
        os.path.join(folder, 'test_embedding_hidden_embedding_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'val_embedding_embedding_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'test_embedding_embedding_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'train_embedding_input_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'test_embedding_ys_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'train_embedding_ys_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'val_embedding_predictions_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'test_embedding_ts_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'train_embedding_ts_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'val_embedding_hidden_embedding_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'test_embedding_predictions_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'test_embedding_name_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'val_embedding_ts_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'train_embedding_predictions_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'val_embedding_input_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'val_embedding_name_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'test_embedding_input_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'train_embedding_name_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'val_embedding_ys_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'train_embedding_embedding_2022-11-16T23:48:34.pkl'),
        os.path.join(folder, 'train_embedding_hidden_embedding_2022-11-16T23:48:34.pkl'),
    ]

    res = {}
    for i in range(len(file_list) ):
        # with Pool(n_processes) as p:
        res[file_list[ i]] = read_pickle_file(file_list[ i])

    train_embedding=np.concatenate(res[os.path.join(folder, 'train_embedding_hidden_embedding_2022-11-16T23:48:34.pkl')])
    train_name=np.concatenate(res[os.path.join(folder, 'train_embedding_name_2022-11-16T23:48:34.pkl')])
    train_predictions=np.concatenate(res[os.path.join(folder, 'train_embedding_predictions_2022-11-16T23:48:34.pkl')])
    train_input=np.concatenate(res[os.path.join(folder, 'train_embedding_input_2022-11-16T23:48:34.pkl')])
    train_ts=np.concatenate(res[os.path.join(folder, 'train_embedding_ts_2022-11-16T23:48:34.pkl')])
    train_ys=np.concatenate(res[os.path.join(folder, 'train_embedding_ys_2022-11-16T23:48:34.pkl')])

    val_embedding=np.concatenate(res[os.path.join(folder, 'val_embedding_hidden_embedding_2022-11-16T23:48:34.pkl')])
    val_name=np.concatenate(res[os.path.join(folder, 'val_embedding_name_2022-11-16T23:48:34.pkl')])
    val_predictions=np.concatenate(res[os.path.join(folder, 'val_embedding_predictions_2022-11-16T23:48:34.pkl')])
    val_input=np.concatenate(res[os.path.join(folder, 'val_embedding_input_2022-11-16T23:48:34.pkl')])
    val_ts=np.concatenate(res[os.path.join(folder, 'val_embedding_ts_2022-11-16T23:48:34.pkl')])
    val_ys=np.concatenate(res[os.path.join(folder, 'val_embedding_ys_2022-11-16T23:48:34.pkl')])
    
    test_embedding=np.concatenate(res[os.path.join(folder, 'test_embedding_hidden_embedding_2022-11-16T23:48:34.pkl')])
    test_name=np.concatenate(res[os.path.join(folder, 'test_embedding_name_2022-11-16T23:48:34.pkl')])
    test_predictions=np.concatenate(res[os.path.join(folder, 'test_embedding_predictions_2022-11-16T23:48:34.pkl')])
    test_input=np.concatenate(res[os.path.join(folder, 'test_embedding_input_2022-11-16T23:48:34.pkl')])
    test_ts=np.concatenate(res[os.path.join(folder, 'test_embedding_ts_2022-11-16T23:48:34.pkl')])
    test_ys=np.concatenate(res[os.path.join(folder, 'test_embedding_ys_2022-11-16T23:48:34.pkl')])

    data= {
        "train_embedding":train_embedding,
         "train_name":train_name,
         "train_predictions":train_predictions,
         "train_input":train_input,
         "train_ts":train_ts,
         "train_ys":train_ys,
         "val_embedding":val_embedding,
         "val_name":val_name,
         "val_predictions":val_predictions,
         "val_input":val_input,
         "val_ts":val_ts,
         "val_ys":val_ys,
         "test_embedding":test_embedding,
         "test_name":test_name,
         "test_predictions":test_predictions,
         "test_input":test_input,
         "test_ts":test_ts,
         "test_ys":test_ys,
        }
    

    train_cols = [k for k in data.keys() if 'train' in k and 'input' not in k]
    # assert all first dimension is the same
    assert all([data[k].shape[0] == data[train_cols[0]].shape[0] for k in train_cols])
    # assert all second dimension is the same
    # assert all([data[k].shape[1] == data[train_cols[0]].shape[1] for k in train_cols])

    train_df = pd.DataFrame({
        'name':data['train_name'],
        # 'ts':data['train_ts'],
        'ys':data['train_ys'].tolist(),
        # 'predictions':data['train_predictions'].tolist(),
        'lstm_embedding':data['train_embedding'].tolist()
    })
    # use name column  as index 
    train_df.set_index('name', inplace=True)
    assert len(train_df.loc[train_df['ys'].isna()]) == 0



    val_cols = [k for k in data.keys() if 'val' in k and 'input' not in k]
    # assert all first dimension is the same
    assert all([data[k].shape[0] == data[val_cols[0]].shape[0] for k in val_cols])

    val_df = pd.DataFrame({
        'name':data['val_name'],
        # 'ts':data['val_ts'],
        'ys':data['val_ys'].tolist(),
        # 'predictions':data['val_predictions'],
        'lstm_embedding':data['val_embedding'].tolist()
    })
    val_df.set_index('name', inplace=True)


    test_cols = [k for k in data.keys() if 'test' in k and 'input' not in k]
    # assert all first dimension is the same
    assert all([data[k].shape[0] == data[test_cols[0]].shape[0] for k in test_cols])

    test_df = pd.DataFrame({
        'name':data['test_name'],
        # 'ts':data['test_ts'],
        'ys':data['test_ys'].tolist(),
        # 'predictions':data['test_predictions'],
        'lstm_embedding':data['test_embedding'].tolist()
    })
    test_df.set_index('name', inplace=True)
    
    return train_df, val_df, test_df


def read_mgrnn_embeddings(FPATH):
    
    data = {}
    data['test_embeddings']=torch.load(FPATH  + '/test_embeddings.pt'  )
    data['test_names']=torch.load(FPATH  + '/test_names.pt'  )
    data['train_embeddings']=torch.load(FPATH  + '/train_embeddings.pt'  )
    data['train_names']=torch.load(FPATH  + '/train_names.pt'  )
    data['val_embeddings']=torch.load(FPATH  + '/val_embeddings.pt'  )
    data['val_names']=torch.load(FPATH  + '/val_names.pt')
    
    
    return  [create_df('train','grnn',data),create_df('val','grnn',data),create_df('test','grnn',data)]

def get_embeddings_df():

    # TODO GLOBAL ENV variables
    FOLDER_PATH_STATS = '/Users/dproios/paper_31dec2022/mimic3-benchmarks/statistical_features'
    FOLDER_PATH_MGR = '/Users/dproios/paper_31dec2022/mimic3-benchmarks/MGRN_embeddings'
    FOLDER_PATH_LSTM='/Users/dproios/paper_31dec2022/mimic3-benchmarks/16_11_2022'
    # FOLDER_PATH_STATS = '/home/dproios/work/create_EHR_gra/statistical_features'
    # FOLDER_PATH_MGR  = '/home/dproios/work/create_EHR_gra/MGRN_embeddings'
    # FOLDER_PATH_LSTM='/home/dproios/work/create_EHR_gra/16_11_2022'
    stat_train_df, stat_val_df, stat_test_df= read_statistical_features(FOLDER_PATH_STATS)
    lstm_train_df, lstm_val_df, lstm_test_df= read_lstm_embeddings(FOLDER_PATH_LSTM)
    mgrn_train_df, mgrn_val_df, mgrn_test_df=read_mgrnn_embeddings(FOLDER_PATH_MGR)

    # train_df = pd.concat([stat_train_df, lstm_train_df,mgrn_train_df], axis=1)
    train_df = stat_train_df.join(lstm_train_df).join(mgrn_train_df)
    val_df = stat_val_df.join(lstm_val_df).join(mgrn_val_df)
    test_df = stat_test_df.join( lstm_test_df).join(mgrn_test_df)

    assert len(train_df.columns) >0
    assert len(val_df.columns) >0
    assert len(test_df.columns) >0

    # assert no cell is nan for each column
    assert all([train_df[col].isna().sum() == 0 for col in train_df.columns])
    assert all([val_df[col].isna().sum() == 0 for col in val_df.columns])
    assert all([test_df[col].isna().sum() == 0 for col in test_df.columns])
    return train_df, val_df, test_df


if __name__ == '__main__':
    get_embeddings_df()