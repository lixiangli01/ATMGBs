
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import math
from sklearn.model_selection import KFold
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = min(1, math.log(lamda/l+1))
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual: # speed up convergence of the training process
            output = output+input
        return output


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        
        
        return layer_inner


class GraphPPIS(nn.Module):
    def __init__(self, nlayers,  nhidden, nclass, dropout, lamda, alpha, variant):
        super(GraphPPIS, self).__init__()
        
        self.lstm=nn.LSTM(512,256,batch_first=True,bidirectional=True,num_layers=4)
        
        self.deep_gcn_esm = deepGCN(nlayers = nlayers, nfeat = 1280, nhidden = nhidden, nclass = nclass,#1280+531
                                dropout = dropout, lamda = lamda, alpha = alpha, variant = variant)
        
        self.deep_gcn_t5 = deepGCN(nlayers = nlayers, nfeat = 2835, nhidden = nhidden, nclass = nclass,#1024+531
                                dropout = dropout, lamda = lamda, alpha = alpha, variant = variant)
        
        
        
        self.deep_gcn_esm_1=deepGCN(nlayers = nlayers, nfeat = 512, nhidden = nhidden, nclass = nclass,
                                dropout = dropout, lamda = lamda, alpha = alpha, variant = variant)
        
        self.deep_gcn_t5_1 = deepGCN(nlayers = nlayers, nfeat = 512, nhidden = nhidden, nclass = nclass,
                                dropout = dropout, lamda = lamda, alpha = alpha, variant = variant)
        
        self.criterion = nn.CrossEntropyLoss() l
        self.optimizer = torch.optim.Adam(self.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        
        self.fc = nn.Sequential(nn.Linear(1024, 256),#256#
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.1),
                                        nn.Linear(256, 64),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.1),
                                        nn.Linear(64, 2))
        self.fc_gcn=nn.Linear(1024, 128)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=531,
            nhead=3#4
        )
        
    def forward(self, esmfea,t5fea, adj,w_embedding):          
        
        t5_x = t5fea.float()
        

        
        
        t5_x=torch.squeeze(t5_x)
        w_in=w_embedding.float()
        w_tr = self.transformer(w_in)
        w_tr=torch.squeeze(w_tr)

        input=torch.cat([t5_x,w_tr],dim=1)

        output_t5 = self.deep_gcn_t5(input, adj)#1024


        
        output=self.fc(output_t5)
        
        return output


###################################################################################################
# dataset
def norm_dis(mx): # from SPROF
    return 2 / (1 + (np.maximum(mx, 4) / 4))

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def load_graph(sequence_name):

    
    #dismap = np.load(Feature_Path + "distance_map/" + sequence_name + ".npy")
    
    dismap = np.load( args.T5_attentionmap_path)
    
    dismap=dismap[:-1,:-1]
    
    
    return dismap#dismap#norm_matrix

def esmembedding(sequence_name):
    fea = np.load( args.esm_features_path)
    
    return fea

def t5embedding(sequence_name):
    fea = np.load( args.T5_features_path )
    fea=fea[:-1]
    return fea

############################################################
import numpy as np
allindex=[]
with open('/home/xli/NABProt/task2_aten/PNA_aaindexs/AAindex.txt', 'r') as f:
    for line in f:
        templist=line.strip().split()
        allindex.append(templist)
allindexnd=np.array(allindex)[:,1:]

aadict={}
for i in range(20):
    key=allindexnd[0,i]
    # print(key)
    values=[float(i) for i in allindexnd[1:,i]]
   
    aadict[key]=np.array(values)

aadict['X']=np.array([0]*531)

def aaindexfea(seq):
    aalist=[]
    for i in seq:
        if i not in 'ARNDCQEGHILKMFPSTWYV':
            aalist.append('X')
        else:
            aalist.append(i)
    seqndarray=np.array([aadict[k] for k in aalist])
    return seqndarray
############################################################
from gensim.models import Word2Vec

def seq_to_kmers1(seq, k=1):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    #seq='XX' + seq + 'XX'
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]

def get_protein_embedding1(model,protein):
    
    vec = np.zeros((len(protein), 128))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

def w1embedding(seq):
    
    model = Word2Vec.load("/home/xli/NABProt/DNAword2vec_1.model")
    vector = get_protein_embedding1(model,seq_to_kmers1(seq))
    #print(vector.shape)
    return vector
#####################################################################
def seq_to_kmers4(seq, k=4):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    seq=seq + 'XXX'
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


def get_protein_embedding4(model,protein):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(protein), 128))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

def w4embedding(seq):
    
    model = Word2Vec.load("/home/xli/NABProt/DNAword2vec_4.model")
    vector = get_protein_embedding4(model,seq_to_kmers4(seq))
    #print(vector.shape)
    return vector

#############################################################
class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['name'].values
        
        self.labels = dataframe['labels'].values
        self.seqs=dataframe['seq'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        seq=self.seqs[index]
        
        label= np.array(list(self.labels[index])).astype('int')#self.labels[index]
        esm_embedding = esmembedding(sequence_name)
        t5_embedding = t5embedding(sequence_name)
        aaindex_fea=aaindexfea(seq)

        
        t5_embedding =np.concatenate([t5_embedding, esm_embedding], axis=1)

        
        w_embedding=aaindex_fea

        

        graph=load_graph(sequence_name)
        
        return sequence_name, esm_embedding,t5_embedding, label , graph,w_embedding 
    
    def __len__(self):
        return len(self.labels)
        

############################################################################################

from torch.autograd import Variable
def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_name, esm_embedding,t5_embedding, label , graph,w_embedding = data
            if torch.cuda.is_available():
                esm_features = Variable(esm_embedding.cuda())
                t5_features = Variable(t5_embedding.cuda())
                w_embedding=Variable(w_embedding.cuda())

                graph = Variable(graph.cuda())
                y_true = Variable(label.cuda())
                #mutsite=Variable(mutsite.cuda())
            else:
                esm_features = Variable(esm_embedding)
                t5_features = Variable(t5_embedding)
                w_embedding = Variable(w_embedding)

                graph = Variable(graph)
                y_true = Variable(label)
                #mutsite=Variable(mutsite)

            #node_features = torch.squeeze(node_features)
            graph = torch.squeeze(graph)
            y_true = torch.squeeze(y_true)

            y_pred = model(esm_features,t5_features, graph,w_embedding)  # y_pred.shape = (L,2)

            # calculate loss
            loss = model.criterion(y_pred, y_true)
            

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_name[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict



LAYER = 8




HIDDEN_DIM = 1024#512
NUM_CLASSES = 2
DROPOUT = 0.1
LAMBDA = 1.5
ALPHA = 0.7
VARIANT = False#True#
LEARNING_RATE = 1E-3 #1E-3#1E-5
WEIGHT_DECAY = 0


SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)


MAP_TYPE = "c"



model = GraphPPIS(LAYER, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
if torch.cuda.is_available():
    model.cuda()


import argparse

parser = argparse.ArgumentParser(description=" nucleic acid-protein Binding Site Prediction")

parser.add_argument('--sequence', type=str, required=True, help='Amino acid sequence (e.g., ACDEFGHIKLMN)')

parser.add_argument('--binding_type', type=str, required=True, help='DNA or RNA')

parser.add_argument('--args.esm_features_path', type=str, required=True, help='Path to esmfeature .npy file')
parser.add_argument('--args.T5_features_path', type=str, required=True, help='Path to T5feature .npy file')
parser.add_argument('--args.T5_attentionmap_path', type=str, required=True, help='Path to T5_attentionmap .npy file')

args = parser.parse_args()

namelist=['tmpname']

seqlist=[]
seqlist=seqlist.append(args.sequence)

labellist=[str('0'*len(seqlist[0]))]  
datadict={'name':namelist,'seq':seqlist,'labels':labellist}
testset=pd.DataFrame(datadict)
#testset=pd.read_csv('/home/xli/NABProt/task2_NABPs/dataprocess/DNArawdataprocess/DNAtest129.csv')

test_loader = DataLoader(dataset=ProDataset(testset), batch_size=1, shuffle=True, num_workers=2)
if args.binding_type=='DNA':
    model.load_state_dict(torch.load('/home/xli/NABProt/PDNAPred/dnamodel/epoch5DNA.pkl'))
else:
    model.load_state_dict(torch.load('/home/xli/NABProt/PDNAPred/rnamodel/epoch5RNA.pkl'))

epoch_loss_avg, valid_true, valid_pred, _  = evaluate(model, test_loader)

print(valid_pred)
