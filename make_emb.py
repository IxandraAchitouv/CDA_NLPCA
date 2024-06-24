# Code writen by I. Achitouv 2024 to generate training/testing datasets
# this code take as input the tweets and write the embeding in two separate files, one for training one for testing - the split bwt data is done by eigencentrality and can be tuned depending on sample required


import numpy as np
from time import process_time
import pandas as pd
#import matplotlib.pyplot as plt 
import pickle as pkl
import random
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('vinai/bertweet-base')
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizert5 = T5Tokenizer.from_pretrained("t5-small")
modelt5 = T5ForConditionalGeneration.from_pretrained("t5-small")

pd.options.mode.chained_assignment = None  # default='warn'
import re
from string import punctuation



def cleaner(text):
  
    text = re.sub('[0-9]+', '', text)
    
    text = re.sub('rt', '', text)
    text = re.sub('RT', '', text)
    text = re.sub('[!@#$]', '', text)
    
    text=re.sub('&amp;','', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text=re.sub(r'http\S+', '', text)
    pattern = r'[^\w\s,]'
    text = re.sub(pattern, '', text)
    return text



def make_time(df,day0,day):
        
        
        #df['timestamp']=pd.to_datetime(df['timestamp'],format='%Y-%m-%d').dt.date
        #df.drop['timestamp']
        
         ## convert (in days)
    df['days']=(day-day0)/ np.timedelta64(1, 'D')
        #df['days'] = (df['timestamp'] - df['days'])/ np.timedelta64(1, 'D')
    return df
    
def embed(text):
    
    return model.encode(text[:249])
    

def timesel(df,D):
    dfsel=df[df['days']==D]
    return dfsel

# Create a function to fit models and make predictions
def predict(model, Xtest):
    return model.predict(Xtest)

# Combine predictions and take the one with highest agreement btw models
def average_predictions(models, Xtest):
    predictions = []
    for model in models:
        preds = predict(model,Xtest)
        predictions.append(preds)
    #predictions=(np.array(predictions).reshape(len(Xtest),len(models)))
    
    res=[]
    for i in range(len(Xtest)):
        l=[]
        for j in range(len(models)):
            l.append(predictions[j][i])
         
        res.append(max(set(l), key=l.count))
       
    return res


pathdata='/home/ixandra/data/tweetsClimat/'

dftemp=pd.read_csv('/home/ixandra/Pclus/data/Gephi_clim29k.csv')
dfank=dftemp[dftemp['eigencentrality']>np.quantile(dftemp['eigencentrality'],0.75)]
dftest=dftemp[(dftemp['eigencentrality']<np.quantile(dftemp['eigencentrality'],0.75)) & (dftemp['eigencentrality']>np.quantile(dftemp['eigencentrality'],0.)) ]

ankuser=dfank['Id'].unique().tolist()
#usertest2=dftest['Id'].unique().tolist()
#Ntest=int(len(usertest2)/1.001)
#random.shuffle(usertest2)
#usertest=usertest2[:Ntest]
ankersall=ankuser #+usertest
print(len(ankersall))

print('len ankers',len(ankuser))
print('len users',len(usertest))


def make_emb(nsample):
    tstart = process_time()
    # select period, users, embed the tweet and write output file
    day0=pd.to_datetime('2022-01-01')
    Nday=30*12
    datelist = pd.date_range(start='1/1/2022', periods=Nday, freq='D').tolist()


    dcol={'timestamp': 'string','user_id': 'int64','text': 'string','retweeted': 'int64','original_author':'int64',
        'proclim': 'float64','contraclim': 'float64'}

    dfall=[]
    for iday in range(0,Nday,nsample):
        d=str(datelist[iday])
        file=pathdata+'all/'+d[:10]+'.csv'
        dftemp=pd.read_csv(file,dtype=dcol,sep=',')
        df2=dftemp.loc[dftemp['user_id'].isin(ankersall)]
        df2=make_time(df2,day0,datelist[iday])
        df2=df2.filter(['days','user_id','text','retweeted','original_author','proclim','contraclim'])

        df2['text']=df2['text'].apply(lambda x: cleaner(x))
        df2['emb']=df2['text'].apply(lambda x: embed(x))
        if iday%nsample==100:
            print('len df',len(df2),'iday=',iday)
        dfall.append(df2)


    dfall=pd.concat(dfall)
    dfall = dfall.dropna()

    dfankers=dfall.loc[dfall['user_id'].isin(ankuser)]
    dfankers = dfankers.sample(frac = 1)
    dfankers.sort_values(by='days', inplace=True)
    dfankers.to_csv('/home/ixandra/Pclus/data/dfankers',index=False)
    print('len dfanker=',len(dfankers))

    dfaver=dfall.loc[dfall['user_id'].isin(usertest)]
    dfaver = dfaver.sample(frac = 1)
    dfaver.sort_values(by='days', inplace=True)
    dfaver.to_csv('/home/ixandra/Pclus/data/dfusers',index=False)
    print('len dfusers=',len(dfaver))




    tstop = process_time()
    print('time to run in sec',round(tstop-tstart,0))
    return dfankers

make_emb(1)   

            


