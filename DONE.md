
# TODO Create graphs run for 

## TODO expertm1 expertm2 expertm3 
for this you need be sure nodes are in the same order as the edges
You need to SOURCE/TARGET of the derived edges if such a thing exists
> python create_expert_edges.py --small_part --working_dataset_path /Users/dproios/paper_31dec2022/mimic3-benchmarks/data_m3benchmark/data/phenotyping

scp -rP 10589 dproios@heg-rl001.hesge.ch:/home/dproios/work/create_EHR_gra/alldfs_categoriesTrue.csv .

python create_expert_edges.py

for lenient put it online 
in create_homogeneous_graph.py
## TODO sigma
This should be easier just you will need to adjust the quantile function in edge strategie


## python create_homogeneous_graph.py --edge_strategy_name random --node_embeddings_type stat  --n_edges 10000
data_e_random_nf_stat_15_12_2022
## python create_homogeneous_graph.py --edge_strategy_name knn_graph --node_embeddings_type stat --k 10
data_e_knn_graph_nf_stat_15_12_2022
## python create_homogeneous_graph.py --edge_strategy_name random --node_embeddings_type lstm  --n_edges 10000
> data_e_random_nf_lstm_15_12_2022
## python create_homogeneous_graph.py --edge_strategy_name knn_graph --node_embeddings_type lstm --k 10
> data_e_knn_graph_nf_lstm_15_12_2022/processed/data.pt

# TODO execute origin code - find origin code 1/2 
## Response 

### rawm3 physionet
wget


### m3benchmark
#### WHY ??? => fix in scp -rP 10589 dproios@heg-rl001.hesge.ch:/home/dproios/work/data/mimic3_benchmark_v0/data/ ./data_m3benchmark
missing length of stay
#### export path to dir
#### debug 

> if hostname heg 
    export PATH_TO_MIMIC-III_CSVs=/home/dproios/work/data/mimiciii/physionet.org/files/mimiciii/1.4
> if m1
    export PATH_TO_MIMIC-III_CSVs=/Users/dproios/paper_31dec2022/mimic3-benchmarks/raw_m3
python -m mimic3benchmark.scripts.extract_subjects {PATH_TO_MIMIC-III_CSVs} data/root/
python -m mimic3benchmark.scripts.validate_events data/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
python -m mimic3benchmark.scripts.split_train_and_test data/root/
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/

### stat 
#### why? because you use it to train graph models and metaembedding

python -um mimic3models.phenotyping.logistic.main --output_dir mimic3models/phenotyping/logistic_regression_results  --data_save_dir mimic3models/phenotyping/data --grid search # -- put the best saved as defaul

 
### LSTM embeddings
> graph1 
train
cd /home/dproios/work/create_EHR_gra/SCEHR/mimic3models/phenotyping 
python dp_190922.py --depth 1 --dropout 0.3 --batch_size 8 --cuda --dim 256 

/home/dproios/work/create_EHR_gra/SCEHR/mimic3models/phenotyping/pytorch_states_2022-11-14 02:53:51/BCE/LSTM.i76.h256.L1.c25.D0.3.BCE+SCL.a0.bs8.wdcy0.epo12.Val-AucMac0.7701.AucMic0.8200.Tst-AucMac0.7700.AucMic0.8192.pt

./pytorch_states_2022-11-14 02:53:51/BCE/LSTM.i76.h256.L1.c25.D0.3.BCE+SCL.a0.bs8.wdcy0.epo12.Val-AucMac0.7701.AucMic0.8200.Tst-AucMac0.7700.AucMic0.8192.pt

> 290922.py create pickles
create embedding files s
> /home/dproios/work/create_EHR_gra/create_homogeneous_graph_4.py 
load embeddings

### Memory GRU Embeddings
>/home/dproios/work/mGRN/mimic3/phenotyping$ 
> python get_mgru_embeddings.py 




# DONE find origin code 1/2
<code>
FOLDER_PATH_STATS = '/Users/dproios/paper_31dec2022/mimic3-benchmarks/statistical_features'
FOLDER_PATH_MGR = '/Users/dproios/paper_31dec2022/mimic3-benchmarks/MGRN_embeddings'
FOLDER_PATH_LSTM='/Users/dproios/paper_31dec2022/mimic3-benchmarks/16_11_2022'
# FOLDER_PATH_STATS = '/home/dproios/work/create_EHR_gra/statistical_features'
# FOLDER_PATH_MGR  = '/home/dproios/work/create_EHR_gra/MGRN_embeddings'
# FOLDER_PATH_LSTM='/home/dproios/work/create_EHR_gra/16_11_2022'
</code>

# DONE 151222 00h52 python create_homogeneous_graph.py --edge_strategy_name random --node_embeddings_type lstm  --n_edges 1_000_000
python read_embeddings.py

added 
  (use "git restore <file>..." to discard changes in working directory)
        modified:   .gitignore
        modified:   DONE.md
        modified:   save_load_chunk.py

        1212dec.md
        HybridHomogeneous.py
        create_homogeneous_graph.py
        custom_layers.py
        edge_strategies.py
        read_embeddings.py

# DONE transport data needed
## files

> m3benchmark
scp -rP 10589 dproios@heg-rl001.hesge.ch:/home/dproios/work/create_EHR_gra/data/phenotyping/  ./data_m3benchmark    

> stat
scp -rP 10589 dproios@heg-rl001.hesge.ch:/home/dproios/work/create_EHR_gra/statistical_features/ .

> lstm
scp -rP 10589 dproios@heg-rl001.hesge.ch:/home/dproios/work/create_EHR_gra/16_11_2022 .

> grn
scp -rP 10589 dproios@heg-rl001.hesge.ch:/home/dproios/work/create_EHR_gra/MGRN_embeddings .

