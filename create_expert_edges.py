import datetime 
import os 
import itertools as it 
import pickle
import math
import pandas as pd 

from config.config import merge_features, orig_rules, rules, rules_normal, all_mappings, args


def get_multi_hot_encoding(alldfs):
        print(' Multi hot encode')
        # new dataframe with multiple columns one for each dist column of alldfs dataframe
        alldfs_multi_hot = pd.DataFrame()
        alldfs_multi_hot['id'] = alldfs.index
        for col in alldfs.columns:
            alldfs_multi_hot[col] = alldfs[col].apply(lambda x: [0 for i in range(len(rules[col]))] if len(x) == 0 else x).tolist()
            alldfs_multi_hot[col] = alldfs_multi_hot[col].apply(lambda x: [1 if i in x else 0 for i in range(len(rules[col]))]).tolist()
            for i in range(len(rules[col])):
                alldfs_multi_hot[col+'_'+str(i)] = alldfs_multi_hot[col].apply(lambda x: x[i])
            alldfs_multi_hot.drop(col, axis=1, inplace=True)
        alldfs_multi_hot.set_index('id', inplace=True)
        return alldfs_multi_hot
        
def exact_match_strategy(spstr):
    fname = f'A_expert_edges_exact{spstr}.pk'
    if not os.path.exists(fname):
        # graph_common_anomalies = False
        # if graph_common_anomalies:
        alldfs= pd.read_csv(f'alldfs_categories{spstr}.csv')
        alldfs=alldfs.set_index(['Unnamed: 0'])
        # select cells of dataframe  whocse  value is empty pd.Series
        alldfs=alldfs.applymap(lambda x: '[]' if (type(x) == str and 'Series' in x) else x)
        alldfs=alldfs.applymap(lambda x: eval(x))

        # for all cells in df if it is in normal category then replace it with nan
        all_normal_categories = [rules_normal[col] for col in rules_normal.keys()]
        alldfs=alldfs.applymap(lambda x: [] if x == [all_normal_categories]  else x)

        # replace nan with empty list
        alldfs=alldfs.applymap(lambda x: [] if x == [math.nan]  else x)

        alldfs_multi_hot = get_multi_hot_encoding(alldfs)
        import datetime
        print('Calculate exact adjacency between episodes')
        print('Current time: ', datetime.datetime.now())

        gs= alldfs_multi_hot.groupby(list(alldfs_multi_hot.columns))
        clusters = list(gs.groups.values())
        expert_edges = pd.Series(clusters).apply(lambda x : [x for x in it.combinations(x, 2)])
        pickle.dump(expert_edges, open(fname, 'wb'))
        print(f'Saved: {fname}')
    print(f'Ar = {fname} loaded')
    Ar = pickle.load(open(fname, 'rb'))
    return Ar

def intersection_strategy(spstr):
    fname = f'A_m2_expert_edges_inter_category{spstr}.pk'
    if not os.path.exists(fname):
        # graph_common_anomalies = False
        # if graph_common_anomalies:
        alldfs= pd.read_csv(f'alldfs_categories{spstr}.csv')
        

        alldfs=alldfs.set_index(['Unnamed: 0'])

        # select cells of dataframe  whocse  value is empty pd.Series
        alldfs=alldfs.applymap(lambda x: '[]' if (type(x) == str and 'Series' in x) else x)
        alldfs=alldfs.applymap(lambda x: eval(x))

        ## alldfs when cell is []  then fill with column normal values 
        alldfs = alldfs.apply(lambda x: x.apply(lambda y: [rules_normal[x.name]] if y == [] else y))
        # alldfs1 = alldfs.apply(lambda x:  rules_normal[x.name], x[0])
        
        # for all cells in df if it is in normal category
        # all_normal_categories = [rules_normal[col] for col in rules_normal.keys()]
        # alldfs=alldfs.applymap(lambda x: [] if x == [all_normal_categories]  else x)

        # replace nan with empty list
        # alldfs=alldfs.applymap(lambda x: [] if x == [math.nan]  else x)

        alldfs_multi_hot = get_multi_hot_encoding(alldfs)
        
        # link all healthy patients to each other
        multihotcols_grouped_by_alldfs_cols = {col:[col+'_'+str(i) for i in range(len(rules[col]))] for col in alldfs.columns}
        normal_columns = [col + '_' + str(rules_normal[col]) for col in alldfs.keys()]
        abnormal_columns = [i for i in alldfs_multi_hot.columns if i not in normal_columns]
        multihotcols_grouped_by_alldfs_abnormal_cols = {}
        for col in abnormal_columns:
            multihotcols_grouped_by_alldfs_abnormal_cols[col.split('_')[0]] = []

        for col in abnormal_columns:
            multihotcols_grouped_by_alldfs_abnormal_cols[col.split('_')[0]].append(col)
        
        
        df = (alldfs_multi_hot[abnormal_columns] == 0)
        df['and'] = df[df.columns[0]]
        for col in df.columns[1:]: 
            df['and'] = df['and'] & df[col]
        healthy_episode_links = [x for x in it.combinations(df[df['and']].index, 2)]
        
        # apply accros column  logical OR 
        alldfs_multi_hot.loc[:, abnormal_columns] = alldfs_multi_hot.loc[:, abnormal_columns].apply(lambda row: row == 0, axis=1)
        for key,cols in multihotcols_grouped_by_alldfs_abnormal_cols.items():
            alldfs_multi_hot[f'{key}_or'] = alldfs_multi_hot[cols[0]]
            for col in cols: 
                print(f'key: {key} col: {col}')
                alldfs_multi_hot[f'{key}_or'] = alldfs_multi_hot[f'{key}_or'] | alldfs_multi_hot[col]
        
        # copy to new df the  columns containing '_or'
        df = alldfs_multi_hot[[col for col in alldfs_multi_hot.columns if '_or' in col]]
        
        clusters = df.groupby(df.columns.tolist(),as_index=False) # .size())    
        print('Current time: ', datetime.datetime.now())
        
        expert_edges = pd.Series(clusters.groups.values()).apply(lambda x : [x for x in it.combinations(x, 2)])
        
        expert_edges_filtered = [edges for edges in expert_edges if edges !=[]]
        expert_edges = list(it.chain.from_iterable(expert_edges_filtered))
        # healthy_episode_links # add
        expert_edges = expert_edges + healthy_episode_links
        print('End time: ', datetime.datetime.now())
        
        
        pickle.dump(expert_edges, open(fname, 'wb'))
        print(f'Saved: {fname}')
    print(f'Ar = {fname} loaded')
    Ar = pickle.load(open(fname, 'rb'))
    return Ar

def intersection_lenient(spstr):
    fname = f'A_m3_expert_edges_inter_category{spstr}.pk'
    if not os.path.exists(fname):
        # graph_common_anomalies = False
        # if graph_common_anomalies:
        alldfs= pd.read_csv(f'alldfs_categories{spstr}.csv')
        alldfs=alldfs.set_index(['Unnamed: 0'])

        # select cells of dataframe  whocse  value is empty pd.Series
        alldfs=alldfs.applymap(lambda x: '[]' if (type(x) == str and 'Series' in x) else x)
        alldfs=alldfs.applymap(lambda x: eval(x))

        ## alldfs when cell is []  then fill with column normal values 
        alldfs = alldfs.apply(lambda x: x.apply(lambda y: [rules_normal[x.name]] if y == [] else y))
        
        alldfs_multi_hot = get_multi_hot_encoding(alldfs)
        
        # link all healthy patients to each other
        normal_columns = [col + '_' + str(rules_normal[col]) for col in alldfs.keys()]
        abnormal_columns = [i for i in alldfs_multi_hot.columns if i not in normal_columns]
        
        multihotcols_grouped_by_alldfs_abnormal_cols = {}
        for col in abnormal_columns:
            multihotcols_grouped_by_alldfs_abnormal_cols[col.split('_')[0]] = []

        for col in abnormal_columns:
            multihotcols_grouped_by_alldfs_abnormal_cols[col.split('_')[0]].append(col)
        
        # import pdb; pdb.set_trace()
        # healthy_episode_links
        df = (alldfs_multi_hot[abnormal_columns] == 0)
        df['and'] = df[df.columns[0]]
        for col in df.columns[1:]: 
            df['and'] = df['and'] & df[col]
        healthy_episode_links = [x for x in it.combinations(df[df['and']].index, 2)]
        print('healthy_episode_links')
        print(len(healthy_episode_links))
        # m3 part
        
        # merge_features
        merge_features_one_hot = {}
        for high_lvl_feat, high_lvl_feat_subcategories in merge_features.items():
            for key,cols in multihotcols_grouped_by_alldfs_abnormal_cols.items():
                # import pdb; pdb.set_trace()
                if key in high_lvl_feat_subcategories:
                    if merge_features_one_hot.get(high_lvl_feat) is None:
                        merge_features_one_hot[high_lvl_feat] = cols
                    else:
                        merge_features_one_hot[high_lvl_feat].extend(cols)
                elif key in merge_features.keys(): 
                    merge_features_one_hot[key] = cols
        
        # alldfs_multi_hot = alldfs_multi_hot.apply(lambda row: row == 0, axis=1)
        alldfs_multi_hot = alldfs_multi_hot.loc[:, abnormal_columns] #.apply(lambda row: row == 0, axis=1)
        
        # m2 part 
        # apply accros column  logical OR
        for key,cols in merge_features_one_hot.items():
            alldfs_multi_hot[f'{key}_or'] = alldfs_multi_hot[cols[0]]
            for col in cols: 
                alldfs_multi_hot[f'{key}_or'] = alldfs_multi_hot[f'{key}_or'] | alldfs_multi_hot[col]
 
        # copy to new df the columns containing '_or'
        df = alldfs_multi_hot[[col for col in alldfs_multi_hot.columns if '_or' in col]]
    
        clusters = df.groupby(df.columns.tolist(),as_index=False) # .size())    
        print('Current time: ', datetime.datetime.now())
        
        expert_edges = pd.Series(clusters.groups.values()).apply(lambda x : [x for x in it.combinations(x, 2)])
        
        expert_edges_filtered = [edges for edges in expert_edges if edges !=[]]
        expert_edges = list(it.chain.from_iterable(expert_edges_filtered))
        # healthy_episode_links # add
        expert_edges = expert_edges + healthy_episode_links
        print('End time: ', datetime.datetime.now())
        
        # pickle.dump(expert_edges, open(fname, 'wb'))
        print(f'Not Saved: {fname}')
        print(f'Number of edges {len(expert_edges)}')
    # print(f'Ar = {fname} loaded')
    # Ar = pickle.load(open(fname, 'rb'))
    # print(len(Ar))
    return expert_edges

strategy_name_map = {
    'exact': exact_match_strategy,
    'intersection': intersection_strategy,
    'lenient':  intersection_lenient
}


def run_strategy(strategy_name, spstr):
    print('Running strategy: ', strategy_name)
    if strategy_name not in strategy_name_map:
        raise ValueError(f'Unknown strategy {strategy_name}')
    return strategy_name_map[strategy_name](spstr)

if __name__ == '__main__':
    run_strategy('exact', args.small_part)
    run_strategy('intersection', args.small_part)
    run_strategy('lenient', args.small_part)