import urllib
import nltk
import glob
from itertools import chain
from nltk.corpus import stopwords
from gensim import corpora, models, utils
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def StopWords(f):
 SWFile = [i for i in f if i.lower() not in stopwords.words('english')] 
 return SWFile

def CreateList(Locale):
 ListOfFiles = []
 for name in glob.glob(Locale):
  ListOfFiles.append(name)
 return ListOfFiles

def findPOS(i):
 if nltk.pos_tag(i)[0][1].startswith('N'):
  return 'n'
 elif nltk.pos_tag(i)[0][1].startswith('V'):
  return 'v'
 elif nltk.pos_tag(i)[0][1].startswith('J'):
  return 'a'
 elif nltk.pos_tag(i)[0][1].startswith('R'):
  return 'r'
 else:
  return ''

def Perform_LDA(i):
 NewFile = [] 
 tokens = [utils.to_utf8(token) for token in utils.tokenize(' '.join(i), lower=True, errors='ignore')] 
 for i in nltk.pos_tag(tokens):
  if findPOS(i) == '':
   NewFile.append(i[0])
  else:
   NewFile.append(wordnet_lemmatizer.lemmatize(i[0], pos = findPOS(i)))
 texts = [[i.encode('utf-8')] for i in NewFile]
 dictionary = corpora.Dictionary(texts)
 corpus = [dictionary.doc2bow(text) for text in texts]
 return corpus, models.ldamodel.LdaModel(corpus=corpus, passes = 30, id2word=dictionary, num_topics=3) 

def main():
 FileCorpus = []
 StopCorpus = [] 
 URL = "C:\Users\NYU\Assignment3\\"
 ToyCorpus = URL + "LDAToy" + "\*"
 FileList = CreateList(ToyCorpus)
 for i, f in enumerate(FileList):
  with open(f) as fl:
   FileCorpus.append(fl.read().splitlines()) 
 for i in FileCorpus:
  Words1 = (word for word in str(i).split() if word.isalpha() and len(word)>1)
  StopCorpus.append(StopWords(Words1))
 for i,f in enumerate(StopCorpus):
  Document = StopCorpus[i]
  OPFile = "Modelled_Topics_" + str(FileList[i][FileList[i].rfind('\\')+1:])
  Corpus, LDA = Perform_LDA(Document)
  LDACorpus = LDA[Corpus] # Assigns the topics to the documents in corpus
  scores = list(chain(*[[score for topic, score in topic] \
                        for topic in [doc for doc in LDACorpus]])) #Score computation for Each Topic Cluster
  threshold = sum(scores)/len(scores)
  TopTopics1 = ' '.join(set([j for i,j in zip(LDACorpus,Document) if i[0][1] > threshold]))
  TopTopics2 = ' '.join(set([j for i,j in zip(LDACorpus,Document) if i[1][1] > threshold])) 
  TopTopics3 = ' '.join(set([j for i,j in zip(LDACorpus,Document) if i[2][1] > threshold])) 
  f = open('C:\Users\NYU\Assignment3\LDAToy\\' + OPFile, "w")
  f.write("Topic #" + "1" + "\n");
  f.write(TopTopics1 + "\n");
  f.write("\n");
  f.write("Topic #" + "2" + "\n");
  f.write(TopTopics2 + "\n");  
  f.write("\n");  
  f.write("Topic #" + "3" + "\n");
  f.write(TopTopics3 + "\n");
  f.write("\n");  
  f.close() 
   
if __name__ == "__main__":
    main()