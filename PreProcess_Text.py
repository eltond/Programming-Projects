#import pyodbc 
import pandas as pd
import nltk
import re
import string
import contractions as cont 
from datetime import datetime
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from sklearn.feature_extraction.text import CountVectorizer
from revoscalepy import RxSqlServerData, rx_data_step

stpwords = nltk.corpus.stopwords.words("english")
newStopWords = ["quarter","yes","billion","million","get","q","us","you","we"]
stpwords.extend(newStopWords)

translator = str.maketrans('','', string.punctuation)

hyprnympaths = []
wnsynsets = []
row_flag = []

startTime = datetime.now()
connection_string="Driver={ODBC Driver 17 for SQL Server};Server=,14330;Database=;Trusted_Connection=Yes"
query="""select top 108000 a.transcriptComponentId,   a.componenttext
from dbo.FactEarningsText a

join dbo.FactEarningsCall b
on a.transcriptComponentId = b.transcriptComponentId

where a.componenttext is not null
and b.transcriptcomponentTypeId not in  (3,1,7)

order by rand(checksum(newid()))"""

ds = RxSqlServerData(sql_query = query, connection_string = connection_string,rows_per_read=10000)
df = rx_data_step(ds)

dfToList = df['componenttext'].tolist()
    
#conn = pyodbc.connect('driver={ODBC Driver 11 for SQL Server};'
#                      'server=;'
#                      'database=;'
#                      'Trusted_Connection=yes;')

#sql = "select top 1000 * from dbo.FactEarningsText a"
    
#data = pd.read_sql(sql,conn)
#dfToList = data['componenttext'].tolist()

class CustomVectorizer(CountVectorizer):
                    
    def build_preprocessor(self):
        
        def Text_Cleanup(f):
                text = f.lower()
                text_contraction = cont.fix(text)
                text_punc = text_contraction.translate(translator)
                text_clean = ' '.join([word for word in text_punc.split() if(len(word.lower())>1)])
                text_sent = ' '.join([i for i in text_clean.split() if(i.isalnum() and not i.isdigit())])
                return text_sent
        return (Text_Cleanup)   
        
    def build_tokenizer(self):
        
        def VecTokenize(f):
            word_tokens = list(nltk.word_tokenize(f))
            return word_tokens 
        return(VecTokenize)
    # overwrite the build_analyzer method, allowing one to
    # create a custom analyzer for the vectorizer
    def build_analyzer(self):
        # create the analyzer that will be returned by this method
        def analyser(doc):
            expand_tokens = []
            expand_Lemms = []
            synsets = []
            # apply the preprocessing and tokenzation steps
            sent_text = nltk.sent_tokenize(doc)
            doc_clean = self.build_preprocessor()(doc)
            tokens = self.build_tokenizer()(doc_clean)
            # use CountVectorizer's _word_ngrams built in method
            # to remove stop words and extract n-grams        
            n_grams = list(set(self._word_ngrams(tokens,self.get_stop_words())))
            for x in n_grams:
                my_regex = r"\b(?=\w)" + re.escape(x) + r"\b(?!\w)"
                matched_sent = [s for s in sent_text if len(re.findall(my_regex, s, re.IGNORECASE)) > 0]
                expand_tokens.append(x)
                expand_Lemms.append(matched_sent)
            Expanded_Token_Lemmd = pd.DataFrame({'Word': expand_tokens,'Sentences': expand_Lemms})
            for i in range(0, len(Expanded_Token_Lemmd.index)):
                for row_len in range(0, len(Expanded_Token_Lemmd.iloc[i]['Sentences'])):
                    x = lesk(Expanded_Token_Lemmd.iloc[i]['Sentences'][row_len],Expanded_Token_Lemmd.iloc[i]['Word'].lower().replace(' ','_'))
                    if not x:
                        synsets.append(Expanded_Token_Lemmd.iloc[i]['Word'].lower())
                    else:
                        synsets.append(str(x)[8:-2])
            return(synsets)
        return(analyser)

custom_vec = CustomVectorizer(ngram_range=(1,3),min_df=0.05,max_df=0.8,stop_words=stpwords)

custom_vec.fit(dfToList)

v1 = custom_vec.vocabulary_
Vocab =  pd.DataFrame.from_dict(v1,orient = "index")
DimVocab = Vocab.reset_index().rename(columns = {0:"VocabKey","index":"Synset"})

for i in range(0, len(DimVocab)):
    try:
        syn = wn.synset(DimVocab.iloc[i]['Synset'])
        for paths in syn.hypernym_paths():
            wnsynsets.append(syn)
            hyprnympaths.append(paths)
            row_flag.append("Synset")
    except:
        wnsynsets.append(DimVocab.iloc[i]['Synset'])
        hyprnympaths.append("No Hypernym Path")
        row_flag.append("n-grams")
DimVocabFinal = pd.DataFrame({'Synsets': wnsynsets,'Hypernym Path':hyprnympaths, 'Keyword Flag':row_flag})

dsout = RxSqlServerData(table = "UtilVocabularyNew", connection_string = connection_string,rows_per_read=10000)
rx_data_step(DimVocabFinal,dsout,overwrite=True)

print(datetime.now() - startTime)
