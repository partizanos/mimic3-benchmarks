from sklearn.metrics import plot_confusion_matrix

import torch_geometric

import joblib 
import pprint 
import optuna            
from optuna import Trial, visualization
from optuna.samplers import TPESampler


from sklearn.metrics import accuracy_score
import sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from sklearn.metrics import coverage_error,confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

from torch import nn
from tqdm import tqdm
import pdb 
import pickle
import sklearn.metrics as metrics
import torch.nn.functional as F
from collections import OrderedDict
import torch 
from datetime import datetime
import time
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
import torch_geometric as pyg
from torch.nn import Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import NeighborSampler
from  torch_geometric.utils import subgraph
from torch import cuda

from custom_layers import LazyLayerNorm, get_layer_impl_from_layer_type,activation_name_implementation
# from torchvision import datasets, transforms
# TIMESTAMP_MILLISECONDS = time.strftime("%d%b_%Y_t%H:%M:%S", time.gmtime()) + str(round(time.time() * 1000))
DATE_TIME=datetime.now().strftime("%d%m%Y-%H%M%S")
DAY_MONTH_YEAR = datetime.now().strftime("%d%m%Y")
RANDOM_SHEET_NUMBER=1234568#1234567
DATASET_NAME = './pyg_graph__nodes_hidden_repr_lstm__edges_anomalyGroups4_bins_4_temperat_oxygenSat_systol_diastol/'
WRITER_NAME = str(DATE_TIME)
writer = None 
# device = 'cpu'
device =torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
import math
last_loss = math.inf
max_patience_count = 1
class myGCN(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 dataset,NUM_HIDDEN_LAYERS,NUM_MLP_LAYERS, layer_type, aggr_fn='add', 
                 dropout_rate=0.0,activation_fn_name='celu',layer_norm_flag=False,**kwargs):
        super().__init__()

        torch.manual_seed(RANDOM_SHEET_NUMBER)
        self.NUM_HIDDEN_LAYERS=NUM_HIDDEN_LAYERS
        self.NUM_MLP_LAYERS=NUM_MLP_LAYERS
        self.layer_norm_flag=layer_norm_flag
        self.dropout_rate=dropout_rate
        layer = get_layer_impl_from_layer_type[layer_type]['impl']
        layer_params = get_layer_impl_from_layer_type[layer_type]['params'] 
        if layer_params == {}:
            self.conv1 = layer(dataset.num_features, hidden_channels)
        else: 
            modified_params = layer_params.copy()
            if 'empty_init' in layer_params:
                self.conv1 = layer()
            elif 'default_convs' in layer_params:
                self.conv1 = layer(dataset.num_features, hidden_channels, **layer_params)
            elif 'hidden_channels' in layer_params:
                del layer_params['hidden_channels']
                self.conv1 = layer(
                    hidden_channels,
                    **layer_params
                )
            else:
                self.conv1 = layer(**layer_params)
        for k in self.conv1.state_dict().keys():
            torch.nn.init.xavier_uniform_(self.conv1.state_dict()[k].reshape(1,-1)).reshape(-1)
        self.layer_norm = torch.nn.LayerNorm(hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels,dataset.num_classes)
        
        torch.nn.init.xavier_uniform(self.lin1.weight)

        self.activation_fn = activation_name_implementation[activation_fn_name]()
        
        layers = OrderedDict()
        for i in range(NUM_HIDDEN_LAYERS):
            print('added graphhidden layer: ', i)
            layers[str(i)] = layer(hidden_channels, hidden_channels)
            for k in layers[str(i)].state_dict().keys():
                torch.nn.init.xavier_uniform_(layers[str(i)].state_dict()[k].reshape(1,-1)).reshape(-1)
        self.hidden_layers = Sequential(layers)
        
        mlp_layers = OrderedDict()
        for i in range(NUM_MLP_LAYERS):
            print('added mlp layer: ', i)

            mlp_layers[str(i)] = torch.nn.Linear(hidden_channels,hidden_channels)
            # import pdb; pdb.set_trace()
            torch.nn.init.xavier_uniform(mlp_layers[str(i)].weight)
            
        self.mlp_layers = Sequential(mlp_layers)
        post_mlp_layers = OrderedDict()
        for i in range(NUM_MLP_LAYERS):
            print('added mlp layer: ', i)

            post_mlp_layers[str(i)] = torch.nn.Linear(hidden_channels,hidden_channels)
            # import pdb; pdb.set_trace()
            torch.nn.init.xavier_uniform(post_mlp_layers[str(i)].weight)
            
        self.post_mlp_layers = Sequential(post_mlp_layers)


    def forward(self, x, edge_index):
        edge_index=edge_index.to(device)

        for i in range(self.NUM_MLP_LAYERS):
                    x = self.mlp_layers[i](x)
                    x = self.activation_fn(x)
                    if self.layer_norm_flag:
                        pass
                        # x = self.layer_norm(x)
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        x = self.conv1(x, edge_index)
        x = self.activation_fn(x)
        # import pdb; pdb.set_trace()
        # x = self.layer_norm(x)

        for i in range(self.NUM_HIDDEN_LAYERS):
            x = self.hidden_layers[i](x, edge_index)
            x = self.activation_fn(x)
            if self.layer_norm_flag:
                x = self.layer_norm(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        for i in range(self.NUM_MLP_LAYERS):
            x = self.post_mlp_layers[i](x)
            x = self.activation_fn(x)
            if self.layer_norm_flag:
                pass
                # x = self.layer_norm(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
    
        x = self.lin1(x)
        
        return x.sigmoid()


import torch.nn as nn 
# from mimic3models.pytorch_models.losses import SupConLoss_MultiLabel, SupNCELoss, CBCE_loss, CBCE_WithLogitsLoss

def train(model, data, params_hparam, **kwargs):
    global max_patience_count
    global last_loss
    LR=params_hparam['LR']
    WD=params_hparam['WD']
    num_epochs= params_hparam['NUM_EPOCHS']
    
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_edges = torch_geometric.utils.subgraph(
                train_idx,           
                edge_index=data.edge_index, 
                relabel_nodes=True, 
                num_nodes=None)[0] 
    # from torch_geometric.loader import NeighborLoader

    # data = Planetoid(path, name='Cora')[0]

    # loader = NeighborLoader(
    #     data,
    #     # Sample 30 neighbors for each node for 2 iterations
    #     num_neighbors=[30] * 2,
    #     # Use a batch size of 128 for sampling training nodes
    #     batch_size=128,
    #     input_nodes=data.train_mask,
    # )
    
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    val_edges = torch_geometric.utils.subgraph(
                val_idx,           
                edge_index=data.edge_index, 
                relabel_nodes=True, 
                num_nodes=None)[0] # num_nodes=data.x.shape[0]
    # mask = torch.zeros(num_nodes, dtype=torch.bool)
    # mask[train_idx] = True
    # edge_mask_source = mask[data.edge_index[0]]
    # edge_mask_target = mask[data.edge_index[1]]
    # edge_mask=torch.logical_and(edge_mask_source, edge_mask_target)
    # subedge_index = data.edge_index[:, edge_mask]

    train_loader = NeighborSampler(
        train_edges, 
        node_idx=train_idx, 
        sizes=[params_hparam['NUM_NODES']], 
        batch_size=params_hparam['batch_size'])
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=LR, weight_decay=WD, amsgrad=False)  # Define optimizer.

    # weights = (1/data.y[[train_idx]].sum(axis=0))
    # criterion_BCE = nn.BCELoss(weights*10e5)
    criterion_BCE = nn.BCELoss()
    
    # normalize_tensor_to01 = lambda x : (x-x.min())/(x.max()-x.min())
    # norm_weights = normalize_tensor_to01(weights)
    # # import pdb; pdb.set_trace()
    # criterion_BCE = nn.BCELoss(norm_weights)
    # criterion_SCL_MultiLabel = SupConLoss_MultiLabel(temperature=0.1)   # temperature=0.01)  # temperature=opt.temp

    def get_loss(y_pre, labels, 
                #  representation, 
                    alpha=0
                    ):
        # CBCE_WithLogitsLoss is more numerically stable than CBCE_Loss when model is complex/overfitting
        # import pdb; pdb.set_trace()
        # >>>>>>>>loss = criterion_BCE(y_pre, labels)
        loss = criterion_BCE(y_pre, labels)
        # if alpha > 0:
        #     if len(representation.shape) == 2:
        #         representation = representation.unsqueeze(1)
        #     scl_loss = criterion_SCL_MultiLabel(representation, labels)
        #     loss = loss + alpha * scl_loss
        return loss
    x, y = data.x, data.y
    # for epoch in tqdm(range(1, num_epochs)):
    epoch = 0
    while(True):
        epoch = epoch + 1
        print('epoch: ', epoch)
        model.train()
        for batch_size, n_id, adjs in tqdm(train_loader):
            # batch_label_weights =  (1/data.y[[n_id]].sum(axis=0))
        
            # sample_importance = (data.y[[n_id]] * 1/batch_label_weights.T).sum(axis=1)
            adjs0 = adjs[0]
            optimizer.zero_grad() # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            y_hat = model(x[n_id],adjs0)
            train_loss = get_loss(y_hat,y[n_id].float())
            # import pdb; pdb.set_trace()
            train_loss.backward()
            optimizer.step()
            # edge_index=edge_index.detach().cpu()#[0]
            del adjs
            # gc.collect()
            torch.cuda.empty_cache()
            # print('Cuda memory allocated: ', torch.cuda.memory_allocated(device))
            # print('Cuda utilized memory: ', torch.cuda.memory_reserved(device))
            # print('Cuda memory cached: ', torch.cuda.memory_cached(device))
        # print('Cuda memory max allocated: ', torch.cuda.max_memory_allocated(device))
        # # print('Cuda max utilized memory: ', torch.cuda.max_memory_reserved(device))
        # print('Cuda max memory cached: ', torch.cuda.max_memory_cached(device))
        # print('Cuda utilization: ', torch.cuda.utilization(device))
        # pdb.set_trace()
            # print('Epoch: {:03d}, train_Loss: {:.5f}'.format(epoch,   train_loss.item()))
        writer.add_histogram('train_loss_'+WRITER_NAME, train_loss, epoch)
        
        
        # val_idx = data.val_mask.nonzero(as_tuple=False).view(-1) 

        # val_loader = NeighborSampler(
        #     subedge_index, 
        #     node_idx=val_idx, 
        #     sizes=[params_hparam['NUM_NODES']], 
        #     batch_size=params_hparam['batch_size'])
        # import pdb; pdb.set_trace()
        print('validation')
        model.eval()
        y_hat = model(x[data.val_mask], val_edges)
        val_loss = get_loss(y_hat, data.y[data.val_mask].float())
        writer.add_histogram('val_loss_'+WRITER_NAME, val_loss, epoch)
        print(
            'Epoch: {:03d}, train_Loss: {:.5f}, val_Loss: {:.5f}'.format(epoch, train_loss.item(), val_loss.item()))
        if val_loss > last_loss: 
            max_patience_count = max_patience_count - 1
            print('Patience decreased from: ', max_patience_count+1, ' to: ', max_patience_count)
        else: 
            last_loss = val_loss
        if max_patience_count == 0:
            break
    writer.add_scalar('val_loss_last_epoch', val_loss)
    return optimizer, epoch, val_loss

import numpy as np
def save_torch_tensor_as_npy(tensor, path):
    np.save(path, tensor.detach().cpu().numpy())

def test(model, data, model_name):
        print('Testing...')
        model.eval()
        with torch.no_grad():
            test_idx= data.test_mask.nonzero(as_tuple=False).view(-1)
            test_edges = torch_geometric.utils.subgraph(
                test_idx,           
                edge_index=data.edge_index, 
                relabel_nodes=True, 
                num_nodes=None)[0]
            out = model(data.x[data.test_mask], test_edges)
            # # import pdb; pdb.set_trace()
            # # out = model(data.x, data.edge_index)
            # # save_torch_tensor_as_npy(out, f'{model_name}_test_results.npy')
            # test_correct = torch.abs(out.cpu() - data.y[data.test_mask].cpu()) < 0.5  # Check against ground-truth labels.
            # test_acc = int(test_correct.sum()) / int(test_correct.shape[0]*test_correct.shape[1])  # Derive ratio of correct predictions.
            # tl = data.y[data.test_mask].cpu().detach().numpy()
            # sc = out.cpu().detach().numpy()
            # sc_copy = sc.copy()
            # sc[test_correct.cpu().detach().numpy() == False] = 1
            # sc[test_correct.cpu().detach().numpy() == True] = 0
            # import pdb; pdb.set_trace()
            # # print(tl.shape, sc.shape)

            # import pdb; pdb.set_trace()
            # out = model(data.x, data.edge_index)
            # save_torch_tensor_as_npy(out, f'{model_name}_test_results.npy')
            test_correct = torch.abs(out.cpu() - data.y[data.test_mask].cpu()) < 0.5  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(test_correct.shape[0]*test_correct.shape[1])  # Derive ratio of correct predictions.
            tl = data.y[data.test_mask].cpu().detach().numpy()
            sc = out.cpu().detach().numpy()
            sc_copy = sc.copy()
            sc[test_correct.cpu().detach().numpy() == False] = 1
            sc[test_correct.cpu().detach().numpy() == True] = 0
            # import pdb; pdb.set_trace()
            # print(tl.shape, sc.shape)

            try:
                print('confusion matrix: ')
                (tn, fp), (fn, tp) = metrics.confusion_matrix(tl.ravel(), sc.ravel())
                print('tn, fp, fn, tp: ', tn, fp, fn, tp)
                sensitivity= tp/(tp+fn)
                print('sensitivity',sensitivity)
                writer.add_scalar('sensitivity',sensitivity)
                specificity = tn/(tn+fp)
                print('specificity',specificity)
                writer.add_scalar('specificity',specificity)
                recall = tp/(tp+fn)
                writer.add_scalar('recall',recall)
                print('recall',recall)
                accuracy = (tp+tn)/(tp+tn+fp+fn)
                writer.add_scalar('accuracy',accuracy)
                print('accuracy',accuracy)
                
                ## add confusion matrix to tensorboard
                writer.add_figure('confusion_matrix', plot_confusion_matrix(tl.ravel(), sc.ravel(), classes=['0','1'], normalize=True), global_step=0)
            except Exception as myexc:
                print('confusion matrix failed: ', myexc)
                # import pdb; pdb.set_trace()
                
                # add confusion matrix
                # writer.add_figure('confusion_matrix', plot_confusion_matrix(tl.ravel(), sc.ravel(), classes=['0','1'], normalize=True), global_step=0)
            # tpr, fpr = metrics.true_positive_rate(tl, sc), metrics.false_positive_rate(tl, sc)
            # sensitivity, specificity = metrics.sensitivity(tl, sc), metrics.specificity(tl, sc)
            # pickle.dump(data.y[data.test_mask].cpu().detach().numpy(),open( "truth_labels.npy", "wb"))
            # pickle.dump(out[data.test_mask].cpu().detach().numpy(),open( "out_scores.npy", "wb"))
            auc_c = metrics.roc_auc_score(tl, sc_copy,average=None)
            aucw = metrics.roc_auc_score(tl, sc_copy,average="weighted")
            auc_micro = metrics.roc_auc_score(tl, sc_copy,average="micro")
            auc_macro = metrics.roc_auc_score(tl, sc_copy,average="macro")
            writer.add_scalar('test_acc', test_acc)
            # writer.add_scalar('auc_c', auc_c)
            # writer.add_scalar('aucw', aucw)
            writer.add_scalar('auc_micro', auc_micro)
            writer.add_scalar('auc_macro', auc_macro)
            # writer.add_scalar('sensitivity', sensitivity)
            # writer.add_scalar()
            
            print('Test Accuracy: {:.4f}'.format(test_acc))
            print(f'AUC: {auc_c}, {aucw}, {auc_micro}, {auc_macro}')            
            return {'test_acc':test_acc, 'auc_c':auc_c, 'aucw':aucw, 'auc_micro':auc_micro, 'auc_macro':auc_macro,}
            # report_results( method = 'GCN', auc=auc, test_acc=test_acc)
            # import pdb; pdb.set_trace()
            # return test_acc,auc,out
            # return test_acc,pred,out


def torch_save_model(model, optimizer, epoch, model_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
    print('model saved in {}'.format(model_path))
    
def run_model(dataset, best_params, layer):
    global writer 
    global WRITER_NAME
    data = dataset[0]
    data.to(device)
    params_arch = { **{
                'hidden_channels':None,
                'dataset':dataset,
                'NUM_HIDDEN_LAYERS':None,
                'layer_type':layer,#trial.suggest_categorical('layer_type', ['ClusterGCNConv','LEConv','ResGatedGraphConv'
                # ]),#layer,
                'aggr_fn':'mean', #
        }, **best_params }
    params_hparam ={**{'NUM_EPOCHS': 11,# trial.suggest_int('NUM_EPOCHS', 5, 10),
        }, **best_params }
    
    FNAME = f'./runs_{DAY_MONTH_YEAR}/{dataset.raw_dir}_{layer}_{DATE_TIME}/'
    print('>>> Summary of the run will be saved in: ', FNAME)
    writer = SummaryWriter(FNAME)
    model = myGCN(**params_arch)
    optimizer, epoch,val_loss  = train(model, data, params_hparam)
    metrics_res = test(model, data,layer)
    return model, metrics_res

def objective(trial: Trial, dataset,layer ) -> float:
        global writer 
        global WRITER_NAME
        data = dataset[0]

        data.to(device)
        params_arch = {
                'hidden_channels':trial.suggest_int('hidden_channels', 1, 512),
                'dataset':dataset,
                'NUM_HIDDEN_LAYERS':trial.suggest_int('NUM_HIDDEN_LAYERS', 0,3),
                'layer_type':layer,#trial.suggest_categorical('layer_type', ['ClusterGCNConv','LEConv','ResGatedGraphConv'
                # ]),#layer,
                'aggr_fn':'add', #
        }
        params_hparam = {
                'LR': trial.suggest_float('LR', 1e-5, 1e-1, log=True),
                'WD': trial.suggest_float('WD', 1e-5, 1e-1, log=True),
                "NUM_NODES": trial.suggest_int('NUM_NODES', 1, 512),
                "batch_size": trial.suggest_int('batch_size', 1, 256),
                'NUM_EPOCHS': 3,# trial.suggest_int('NUM_EPOCHS', 5, 10),
        }
        # WRITER_NAME='runs_231022_ResGatedGraphConv_hparam/'+ '_'.join(
        #     [str(k)+'_'+str(v) for k,v in params_hparam.items()]) + '_'.join(
        #     [str(k)+'_'+str(v) for k,v in params_arch.items()])
        WRITER_NAME='runs_301022/'+ layer + '_'.join(
            [str(k)+'_'+str(v) for k,v in params_hparam.items()]) + '_'.join(
            [str(k)+'_'+str(v) for k,v in params_arch.items()])

        print('Using writer name: ',WRITER_NAME)
        writer = SummaryWriter(WRITER_NAME)
        pprint.pprint(params_hparam)
        pprint.pprint(params_arch)
        
         
        # model = myGCN(hidden_channels=HC, dataset=dataset, NUM_HIDDEN_LAYERS=0, layer_type=layer, aggr_fn=aggfn)
        model = myGCN(**params_arch)
        optimizer, epoch,val_loss  = train(model, data, params_hparam)
        #cross_val_score(model, cv=3, n_jobs=2, verbose=1, error_score=accuracy_score).mean()
        print('# objective logs Best params ')
        # print(study.best_trial.params)
        return val_loss
