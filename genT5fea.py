
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import time
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('/home/xli/NABProt/task2/diff_fea/t5_fea/set_file', do_lower_case=False)
model = T5EncoderModel.from_pretrained("/home/xli/NABProt/task2/diff_fea/t5_fea/set_file")

model.to(device)

def extract_fea(mystr):
    sequence_examples=[]
    sequence_examples.append(mystr)
    #sequence_examples = ["PRTEINO", "SEQWENCE"]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)
    emb_0 = embedding_rpr.last_hidden_state[0, :]  # shape (7 x 1024)
    #print(emb_0)
    return emb_0.detach().cpu().numpy()




trainset=pd.read_csv('/home/xli/NABProt/task2_NABPs/dataprocess/RNA161dataprocess/RNAtest161.csv')
print(trainset)
namelist=list(trainset['name'])
seqlist=list(trainset['seq'])
a=time.time()
for i in range(len(namelist)):   
    
    output = extract_fea(seqlist[i])
    dir='/home/xli/NABProt/task2_NABPs/genfea/RNA545_161fea/rna_t5_fea/'+namelist[i]+'.npy'
    np.save(dir,output)
    print(output.shape,namelist[i])
    # print(123)
b=time.time()
print(b-a)




