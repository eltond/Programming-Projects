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

wrd=[]
synsets = []
defn = []
full_sent = []
hyprnympaths = []
expand_tokens = []
expand_Lemms = []
wnsynsets = []

def flattenfile(f):
    flattened_list = []
    for x in f:
        flattened_list.append(x)
    return flattened_list

#Text = ['I really really love love love artificial intelligence and robotics']
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

#sql = "select top 5 * from dbo.FactEarningsText a"
    
#data = pd.read_sql(sql,conn)
#dfToList = data['componenttext'].tolist()


FileCompressed = flattenfile(dfToList)
sent_text = nltk.sent_tokenize(' '.join(FileCompressed))


class CustomVectorizer(CountVectorizer):
                    
    def build_preprocessor(self):
        
        def Text_Cleanup(f):
                text = f.lower()
                text_contraction = cont.fix(text)
                text_punc = text_contraction.translate(translator)
                text_clean = ' '.join([word for word in text_punc.split() if(len(word.lower())>1 and word.lower() not in stpwords)])
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
        
        # load stop words using CountVectorizer's built in method
        preprocess = self.build_preprocessor()
        Tokenize = self.build_tokenizer()
        # create the analyzer that will be returned by this method
        def analyser(doc):
            
            # apply the preprocessing and tokenzation steps
            doc_clean = preprocess(doc)
            tokens = Tokenize(doc_clean)
            #tokens = lemmatizer(doc_clean)
            lemmatized_tokens = [token for token in tokens]
            # use CountVectorizer's _word_ngrams built in method
            # to remove stop words and extract n-grams        
            n_grams = self._word_ngrams(lemmatized_tokens)
            for x in n_grams:
                my_regex = r"\b(?=\w)" + re.escape(x) + r"\b(?!\w)"
                matched_sent = [s for s in sent_text if len(re.findall(my_regex, s, re.IGNORECASE)) > 0]
                expand_tokens.append(x)
                expand_Lemms.append(matched_sent)
            Expanded_Token_Lemmd = pd.DataFrame({'Word': expand_tokens,'Sentences': expand_Lemms})
            for i in range(0, len(Expanded_Token_Lemmd)):
                for row_len in range(0, len(Expanded_Token_Lemmd.iloc[i]['Sentences'])):
                    x = lesk(Expanded_Token_Lemmd.iloc[i]['Sentences'][row_len],Expanded_Token_Lemmd.iloc[i]['Word'].lower().replace(' ','_'))
                    if not x:
                        synsets.append("No Synset")
                    else:
                        synsets.append(str(x)[8:-2])
            return(synsets)
        return(analyser)


custom_vec = CustomVectorizer(ngram_range=(1,2),min_df=8,stop_words=stpwords)

custom_vec.fit(dfToList)

v1 = custom_vec.vocabulary_
Vocab =  pd.DataFrame.from_dict(v1,orient = "index")
DimVocab = Vocab.reset_index().rename(columns = {0:"VocabKey","index":"Word"})

for i in range(0, len(DimVocab)):
    try:
        syn = wn.synset(DimVocab.iloc[i]['Word'])
        for paths in syn.hypernym_paths():
            wnsynsets.append(syn)
            hyprnympaths.append(paths)    
    except:
        wnsynsets.append("No Synset")
        hyprnympaths.append("No Hypernym Path")
DimVocabFinal = pd.DataFrame({'Synsets': wnsynsets,'Hypernym Path':hyprnympaths})
DimVocabFinal['VocabKey'] = range(1, 1+len(DimVocabFinal))

dsout = RxSqlServerData(table = "UtilVocabularyNew", connection_string = connection_string,rows_per_read=10000)
rx_data_step(DimVocabFinal,dsout,overwrite=True)

print(datetime.now() - startTime)

