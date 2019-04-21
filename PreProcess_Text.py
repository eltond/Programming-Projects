import pyodbc 
import pandas as pd
import nltk
import re
import string
import contractions as cont 
from datetime import datetime
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.wsd import lesk
from sklearn.feature_extraction.text import CountVectorizer
from revoscalepy import RxSqlServerData, rx_data_step

translator = str.maketrans('','', string.punctuation)

stpwords = nltk.corpus.stopwords.words("english")
newStopWords = ["quarter","yes","billion","million","get","q","us","you","we"]
stpwords.extend(newStopWords)

lemmatizer = WordNetLemmatizer()

LemmKey = pd.DataFrame(columns= ["Token","Lemmed"])
wsd_words = []
wsd_sent = []
full_words = []
full_sent = []
full_rwords = []
wrd=[]
wsd_rwords = []
synsets = []
defn = []
hyprnympaths = []
expand_tokens = []
expand_Lemms = []
expand_sent = []

#Taken from BuildVocab.py
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN
#lemmatizer.lemmatize('going', wordnet.VERB)

def lemm_tokens(TokPos, lemmatizer):
    lemmed = []
    for i,j in TokPos:
        lemmed.append(lemmatizer.lemmatize(i,get_wordnet_pos(j)))
    return lemmed

def flattenfile(f):
    flattened_list = []
    for x in f:
        flattened_list.append(x)
    return flattened_list

def WordTokenize(f):
    filtered_words = ' '.join([word for word in f.split() if(len(word.lower())>1 and word.lower() not in stpwords)])
    word_tokens = list(set(nltk.word_tokenize(filtered_words)))
    Tags = nltk.pos_tag(word_tokens)
    
    lemwords = lemm_tokens(Tags, lemmatizer)
    
    return word_tokens,lemwords

def VecTokenize(f):
    filtered_words = ' '.join([word for word in f.split() if(len(word.lower())>1 and word.lower() not in stpwords)])
    word_tokens = list(set(nltk.word_tokenize(filtered_words)))
    Tags = nltk.pos_tag(word_tokens)
    
    lemwords = lemm_tokens(Tags, lemmatizer)
    
    return lemwords

def Text_Cleanup(f):
    sent_contraction = cont.fix(f)
    sent_punc = sent_contraction.translate(translator)
    clean_sent = ' '.join([i for i in sent_punc.split() if(i.isalnum() and not i.isdigit())])
    return clean_sent
    
def SynSet_Generate(word,sent):
    return lesk(sent, word)
    
def main():
    startTime = datetime.now()
    connection_string="Driver=SQL Server;Server=(local);Database=TextAnalytics;Trusted_Connection=True"
    query="""select top 108000 a.transcriptComponentId,   a.componenttext
    from dbo.FactEarningsText a

    join dbo.FactEarningsCall b
    on a.transcriptComponentId = b.transcriptComponentId

    where a.componenttext is not null
    and b.transcriptcomponentTypeId not in  (3,1,7)

    order by rand(checksum(newid()))"""

    ds = RxSqlServerData(sql_query = query, connection_string = connection_string,rows_per_read=10000)
    df = rx_data_step(ds)
     
    #conn = pyodbc.connect('driver={ODBC Driver 11 for SQL Server};'
    #                      'server=MRICZDADSAPD005;'
    #                      'database=TextAnalytics;'
    #                      'Trusted_Connection=yes;')

    #sql = "select top 5 * from dbo.FactEarningsText a"
    
    #data = pd.read_sql(sql,conn)
    
    #dfToList = data['componenttext'].tolist()
    
    dfToList = df['componenttext'].tolist()
    
    FileCompressed = flattenfile(dfToList)
    sent_text = nltk.sent_tokenize(' '.join(FileCompressed))
    
    for sentence in sent_text:
        sent_cleaned = Text_Cleanup(sentence)
        word,lemms = WordTokenize(sent_cleaned)
        wsd_rwords.append(word)
        wsd_words.append(lemms)
        wsd_sent.append(sent_cleaned) 
    Token_Lemmd = pd.DataFrame({'Token': wsd_rwords,'Lemmed': wsd_words, 'Sentence': wsd_sent})
    for i in range(0, len(Token_Lemmd)):
        for row_len in range(0, len(Token_Lemmd.iloc[i]['Token'])):
            expand_tokens.append(Token_Lemmd.iloc[i]['Token'][row_len])
            expand_Lemms.append(Token_Lemmd.iloc[i]['Lemmed'][row_len])
            expand_sent.append(Token_Lemmd.iloc[i]['Sentence'])
    Expanded_Token_Lemmd = pd.DataFrame({'Token': expand_tokens,'Lemmed': expand_Lemms, 'Sentence': expand_sent})   
    for i in range(0, len(Expanded_Token_Lemmd)):
        x = SynSet_Generate(Expanded_Token_Lemmd.iloc[i]['Lemmed'],Expanded_Token_Lemmd.iloc[i]['Sentence'])
        if not x:
            wrd.append(Expanded_Token_Lemmd.iloc[i]['Lemmed'])
            synsets.append("No Synset")
            defn.append("No Definition")
            hyprnympaths.append("No Hypernym Path")
        else:
            for paths in x.hypernym_paths():
                wrd.append(Expanded_Token_Lemmd.iloc[i]['Lemmed'])
                synsets.append(x)
                defn.append(x.definition())
                hyprnympaths.append(paths)
    FinalSet = pd.DataFrame({'Words': wrd,'Synsets': synsets,'Definition':defn,'Hypernym Path':hyprnympaths})
    
    FinalSet = FinalSet[['Words','Synsets','Hypernym Path']]

    #Add column rename?
    
    FinalSet.drop_duplicates(inplace=True) 

    FinalSet = FinalSet.assign(VocabKey=[i for i in range(len(FinalSet))])[['VocabKey'] + FinalSet.columns.tolist()]
    
    FinalSet.set_index('VocabKey')
    
    v = CountVectorizer(preprocessor=Text_Cleanup,tokenizer=VecTokenize,analyzer="word",ngram_range=(1,3),min_df=8,stop_words=stpwords)

    response = v.fit(dfToList)

    #Below commented lines seem to create the DimVocab Table
    #v1 = v.vocabulary_
    #Vocab =  pd.DataFrame.from_dict(v1,orient = "index")
    #Vocab = Vocab.rename(columns = {0:"VocabKey"})
    #Vocab["Word"] = Vocab._get_axis(0)

    #dsout = RxSqlServerData(table = "UtilVocabularyNew", connection_string = connection_string,rows_per_read=10000)
    #rx_data_step(Vocab,dsout,overwrite=True)

    Dat = Expanded_Token_Lemmd[['Token','Lemmed']].drop_duplicates() 
    
    global LemmKey
    
    Dat = (pd.merge(Dat, LemmKey, on=Dat.columns.tolist(), how="outer", indicator=True)
        .query("_merge == 'left_only'")
        .drop('_merge', 1)    
        )

    LemmKey = LemmKey.append(Dat,ignore_index=True,sort = True)

    dsout = RxSqlServerData(table = "UtilVocabularyNew", connection_string = connection_string,rows_per_read=10000)
    rx_data_step(FinalSet,dsout,overwrite=True)  

    dsout = RxSqlServerData(table = "UtilLemmateNew", connection_string = connection_string,rows_per_read=10000)
    rx_data_step(LemmKey,dsout,overwrite=True)

    print(datetime.now() - startTime)


main()