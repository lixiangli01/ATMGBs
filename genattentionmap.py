
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import time
import pandas as pd


device = torch.device( 'cpu')#'cuda:0' if torch.cuda.is_available() else

tokenizer = T5Tokenizer.from_pretrained('/home/xli/NABProt/task2/diff_fea/t5_fea/set_file', do_lower_case=False)
model = T5EncoderModel.from_pretrained("/home/xli/NABProt/task2/diff_fea/t5_fea/set_file")

model.to(device)

def extract_fea(mystr):
    sequence_examples=[]
    sequence_examples.append(mystr)
    
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask,output_attentions=True)
    
    atten_map_lastlayer=embedding_rpr.attentions[23][0]#torch.Size([32, 1831, 1831])
    atten_map_lastlayer=torch.mean(atten_map_lastlayer,dim=0)
    return atten_map_lastlayer.detach().cpu().numpy()




trainset=pd.read_csv('/home/xli/NABProt/task2_NABPs/dataprocess/RNA161dataprocess/RNAtest161.csv')
print(trainset)
namelist=list(trainset['name'])
seqlist=list(trainset['seq'])
a=time.time()
n=0

for i in range(len(namelist)):   
    # print(tmpfea)
    output = extract_fea(seqlist[i])


    n+=1
    print(output.shape,namelist[i],n)
    # print(123)
b=time.time()
print(b-a)




