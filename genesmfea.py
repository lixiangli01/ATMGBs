import torch
import esm
import numpy as np
import pandas as pd
# Load ESM-2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


def genfeature_esm(seq):
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    
    esmfea=token_representations[0][1:-1].cpu().numpy()
    return esmfea

def split_sequence(sequence, segment_length=1022):
    num_segments = len(sequence) // segment_length
    segments = [sequence[i * segment_length: (i + 1) * segment_length] for i in range(num_segments)]
    if len(sequence) % segment_length != 0:
        segments.append(sequence[num_segments * segment_length:]) 
    array_list=[]
    for i in segments:
        array_list.append(genfeature_esm(i))

    result_array = np.concatenate(array_list, axis=0)
    return result_array



trainset=pd.read_csv('/home/xli/NABProt/task2_NABPs/dataprocess/RNA161dataprocess/RNAtest161.csv')

print(trainset)

namelist=list(trainset['name'])
seqlist=list(trainset['seq'])




for i in range(len(namelist)):
    
    if len(seqlist[i])<=1024:
        tmpfea=genfeature_esm(seqlist[i])
    else:
        tmpfea=split_sequence(seqlist[i], segment_length=1022)
        
    
    dir='/home/xli/NABProt/task2_NABPs/genfea/RNA545_161fea/rna_esm_fea/'+namelist[i]+'.npy'
    np.save(dir,tmpfea)
