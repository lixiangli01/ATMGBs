import pandas as pd

# testfile=open('/home/xli/NABProt/task2_seq/rawdata/DNA/DNA-129_Test.txt','r')
# testfile=open('/home/xli/NABProt/task2_seq/rawdata/DNA/DNA-181_Test.txt','r')
# testfile=open('/home/xli/NABProt/task2_seq/rawdata/DNA/DNA-129_Test.txt','r')
# testfile=open('/home/xli/NABProt/task2_seq/rawdata/DNA/DNA-129_Test.txt','r')
#testfile=open('/home/xli/NABProt/task2_NABPs/rawdata/RNA/RNA-117_Test.txt','r')
#testfile=open('/home/xli/NABProt/task2_NABPs/rawdata/AB/AB-1011_Train.txt','r')
#testfile=open('/home/xli/NABProt/task2_NABPs/rawdata/MN/MN-144_Test.txt','r')
#testfile=open('/home/xli/NABProt/task2_NABPs/rawdata/RNA161/TR545.txt','r')
testfile=open('/home/xli/NABProt/task2_NABPs/rawdata/RNA161/TE161.txt','r')
allline=[]
for i in testfile:
    # if i[0]=='>':
    #     print(i)
    allline.append(i.strip())
# print(allline)

namelist=[]
seqlist=[]
labellist=[]

for i in range(len(allline)):
    if allline[i][0]=='>':
        namelist.append(allline[i][1:])
        seqlist.append(allline[i+1])
        labellist.append(allline[i+2])
# print(len(namelist))
# print(len(seqlist))
# print(len(labellist))
datadict={'name':namelist,'seq':seqlist,'labels':labellist}
# print(datadict)
df=pd.DataFrame(datadict)
# print(df)

df.to_csv('/home/xli/NABProt/task2_NABPs/dataprocess/RNA161dataprocess/RNAtest161.csv')
