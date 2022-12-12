from concurrent.futures import ProcessPoolExecutor as concPool
import shutil
import multiprocessing as mp
import os 
import pandas as pd

# class NoDaemonProcess(mp.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)

# # We sub-class mp.pool.Pool instead of mp.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(mp.Pool):
#     Process = NoDaemonProcess

def save_splitted_csv(reading_path, writing_folder,chunksize=10000):
    
    path_last_file_without_ext = reading_path.split('/')[-1].split('.')[0]
    print('Reading file', path_last_file_without_ext)
    
    csv_chunk_fname_pattern = path_last_file_without_ext + 'chunk'


    if any(csv_chunk_fname_pattern in file for file in os.listdir(writing_folder)):
        print('Chunk files already exist for' + writing_folder)
    else:
        print('Chunking csv file ...', path_last_file_without_ext, 'and WRITING FILE', writing_folder)
        for i, chunk in enumerate(pd.read_csv(reading_path, chunksize=chunksize)):
            writting_fname= path_last_file_without_ext + 'chunk'
            if i % 100 == 0: print(writting_fname)
            chunk.to_csv(os.path.join(writing_folder, writting_fname), index=False)

def load_split_csv( arguments_list):
    all_chunks = []
    for reading_path, writing_folder in arguments_list:
        reading = reading_path
        ### 
        path_last_file_without_ext = reading_path.split('/')[-1].split('.')[0]
        csv_chunk_fname_pattern = path_last_file_without_ext + '_chunk'
        chunk_files = [file for file in os.listdir(writing_folder) if csv_chunk_fname_pattern in file]

        chunk_files=sorted(chunk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        all_chunks.append(chunk_files)
    all_chunks1 = [item for sublist in all_chunks for item in sublist]

    print('Number of chunks:', len(chunk_files))
    num_cpus=mp.cpu_count()
    ac2=[os.path.join(os.getcwd(),writing_folder, f) for f in all_chunks1]
    # print('files:', ac2)
    with mp.Pool(num_cpus) as p:
        dfs = p.map(pd.read_csv, ac2)
    return dfs


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # shutil.copy2('test.csv', 'test/test.csv')
    arguments_list = list(zip(
        [os.path.join('raw_m3',f) for f in os.listdir('raw_m3') if 'csv' in f], 
        ['test']*len([f for f in os.listdir('raw_m3') if 'csv' in f])
    ))
    # zip(os.listdir('raw_m3'),['test']*len(os.listdir('raw_m3')))
    with mp.Pool(4) as p:
        p.starmap(save_splitted_csv, arguments_list)
    # df1 = load_split_csv('test')
    # df = pd.read_csv('test')
    
    dfs = {}
    print('##############')
    print('TIME CURRENT CHUNK')
    import datetime
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    dfs = load_split_csv(arguments_list)
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    
    # [f[0] for f in arguments_list]
    import pdb; pdb.set_trace()