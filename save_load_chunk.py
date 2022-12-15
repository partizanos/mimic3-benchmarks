"""
This script is used to save and load chunked csv files, which are used to train the model.
It manages to laod all the files in 2 seconds.
"""
from tqdm import tqdm 
import datetime
from concurrent.futures import ProcessPoolExecutor as concPool
import shutil
import multiprocessing as mp
import os 
import pandas as pd
import argparse


args = argparse.ArgumentParser()
# arg wring_folder
args.add_argument('--writing_folder', type=str, default='writing_folder')
args.add_argument('--INPUT_FOLDER', type=str, default='raw_m3')
args.parse_args()




def save_splitted_csv(reading_path, writing_folder,chunksize=10000):
    path_last_file_without_ext = reading_path.split('/')[-1].split('.')[0]
    for i, chunk in tqdm(enumerate(pd.read_csv(reading_path, chunksize=chunksize))):
        writting_fname= path_last_file_without_ext + 'chunk' + str(i) + '.csv'
        chunk.to_csv(os.path.join(writing_folder, writting_fname), index=False)

def chunkFile2df( arguments_list):
    all_chunks = []
    for _, writing_folder in arguments_list:
        chunk_files = sorted(
            [file for file in os.listdir(writing_folder) if file.endswith('.csv')],
            key=lambda x: int(x.split('chunk')[-1].split('.')[0])
        )
        all_chunks.append(chunk_files)
    all_chunks1_bkp = [item for sublist in all_chunks for item in sublist]
    all_chunks1 = [item for sublist in all_chunks for item in sublist]

    print('Number of chunks:', len(chunk_files))
    # num_cpus=mp.cpu_count()
    ac2=[os.path.join(os.getcwd(),writing_folder, f) for f in all_chunks1]
    with mp.Pool(4) as p:
        dfs = p.map(pd.read_csv, ac2)
    return dfs, ac2, all_chunks1_bkp

def save_intact_csv(dfs, names, writing_folder):
    print('save')
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    for df, name in zip(dfs, names):
        fpath = os.path.join(os.path.join(writing_folder, 'intact'), name)
        df.to_csv(fpath, index=False)
        print('saved', fpath)
        print(datetime.datetime.now().strftime("%H:%M:%S"))



writing_folder = args.parse_args().writing_folder
INPUT_FOLDER = args.parse_args().INPUT_FOLDER

os.makedirs(writing_folder, exist_ok=True)

# shutil.copy2('test.csv', 'test/test.csv')
arguments_list = list(zip(
    [os.path.join(INPUT_FOLDER,f) for f in os.listdir(INPUT_FOLDER) if 'csv' in f], 
    ['writing_folder']*len([f for f in os.listdir(INPUT_FOLDER) if 'csv' in f])
))
print(datetime.datetime.now().strftime("%H:%M:%S"))

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    # if writing_folder empty
    if len(os.listdir(writing_folder)) == 0: 
        print('Recreated in', writing_folder)
        with mp.Pool(4) as p: p.starmap(save_splitted_csv, arguments_list)
    
    print('load')
    time_start = datetime.datetime.now()
    dfs, names, hierarchy = chunkFile2df(arguments_list)
    print('loaded in', datetime.datetime.now() - time_start)

    print('Merge chunks to one file ')
    # group dfs according to hierarchy [[df1, df2, df3], [df4, df5, df6]
    dfs_grouped = [dfs[i:i+len(hierarchy[0])] for i in range(0, len(dfs), len(hierarchy[0]))]

    # concat groups of dfs 
    dfs = [pd.concat(dfs_group) for dfs_group in dfs_grouped]
    # print len(dfs)
    print('Number of files:', len(dfs))

    
    # save_intact_csv(dfs, names, writing_folder)    
