import glob
import collections
import math
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as UPGMA
from collections import defaultdict

def CreateList(Locale):
    ListOfFiles = []
    for name in glob.glob(Locale):
        ListOfFiles.append(name)
    return ListOfFiles

def StopWords(Doc):
    with open('C:\Users\NYU\Desktop\A31\stopwords.txt') as f:
        F =''.join(f.read().split()[2])
    StopWords = [i for i in F.replace(",", "").replace('"', " ").replace('  ', " ").split()]
    SWFile = ' '.join([i for i in Doc if i.lower() not in StopWords])     
    return SWFile

def Calculate_CH(i,j,s):
    SE = []
    PjX = []
    PiX = []
    FrequencyCorpusi = collections.defaultdict(float)
    FrequencyCorpusj = collections.defaultdict(float)
    FrequencyCorpuss = collections.defaultdict(float)
    Leni =  float(len(i.split()))
    Lenj =  float(len(j.split()))
    Lens =  float(len(s.split()))            
    for k in i.split():
        FrequencyCorpusi[k] +=float(1)
    for key, value in FrequencyCorpusi.items():
        FrequencyCorpusi[key] = value/Leni
    for l in j.split():
        FrequencyCorpusj[l] +=float(1)
    for key,value in FrequencyCorpusj.items():
        FrequencyCorpusj[key] = value/Lenj
    for m in s.split():
        FrequencyCorpuss[m] +=float(1)
    for key,value in FrequencyCorpuss.items():
        FrequencyCorpuss[key] = value/Lens
    for key in FrequencyCorpusi:
        SE.append(FrequencyCorpusi[key]*math.log(FrequencyCorpusi[key],2))
        PiX.append(((0.99)*(FrequencyCorpusi[key])) + ((0.01)*(FrequencyCorpuss[key]))) #PiX for Kullback–Leibler divergence 
        PjX.append(((0.99)*(FrequencyCorpusj.get(key,0.0))) + ((0.01)*(FrequencyCorpuss[key]))) #PiX for Kullback–Leibler divergence 
    ShannonEntropy = 0 - sum(SE) #Shannon Entropy - H(Xi)
    Qij = 0 - sum([a * math.log(b,2) for a,b in zip(PiX,PjX)]) #Kullback–Leibler Divergence - Q(i||j)
    return (ShannonEntropy/Qij)
    
def main(): 
    URL = 'C:\Users\NYU\Desktop\\'
    ListofInputFiles = URL + "A31" + "\*"
    DendogramImage = "C:\Users\NYU\Desktop\Dendogram" + ".png"
    FieldNames = []
    ReadGroups = []
    CulturalHoleMatrix = [] 
    ArraysOfFileReads = []
    CleanedContent = defaultdict()
    FileContent = defaultdict()
    CulturalHole = defaultdict(list)
    Preprocessed = defaultdict(list)
    FileReads = CreateList(ListofInputFiles)
    with open(FileReads[1]) as f:
        for l in f:
            FieldNames.append(l.strip().split(",")[1])    
    with open(FileReads[0]) as f:
        for l in f:
            ArraysOfFileReads.append(l.strip().split("\t"))
    for i in range(len(ArraysOfFileReads)):
        FileContent[ArraysOfFileReads[i][0]] = ArraysOfFileReads[i][1]
    with open(FileReads[2]) as f1:
        for l in f1:
            ReadGroups.append(l.strip().split("\t"))
        NewList = ReadGroups[1:]
    for i,f in enumerate(NewList[1:]):
        if FileContent[NewList[i][0]] == 'null':
            continue
        else:
            Preprocessed[NewList[i][1]].append(FileContent[NewList[i][0]]) 
    for k,v in Preprocessed.items():
        Words = (word for word in str(v).split() if word.isalpha() and len(word)>1) #Remove all Single Letter & Alpha-Numeric Enteries
        CleanedContent[k] = StopWords(Words)
    KeyList = sorted([int(i) for i in CleanedContent])
    for i in itertools.product(KeyList, repeat=2):
        Writer = str(i[0])
        Reader = str(i[1])
        CulturalHole[i[0]].append(1 - Calculate_CH(CleanedContent[Writer],CleanedContent[Reader],(CleanedContent[Writer] + " " + CleanedContent[Reader])))
    CulturalHoleMatrix = [CulturalHole[key] for key in CulturalHole]
    UPGMAMatrix = np.array(CulturalHoleMatrix)
    #UPGMA Clustering 
    UPGMACluster = UPGMA.average(UPGMAMatrix) 
    fig = plt.figure(figsize=(20,10))  
    plt.title("Document Jargon Distance/Relation")
    #Dendogram Plotting
    UPGMA.dendrogram(UPGMACluster, labels=np.array(FieldNames))
    plt.xlabel("Group Names")
    plt.savefig(DendogramImage)    

   
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total Runtime: %s Seconds" % round((time.time()-start_time)))   