# code writen by I. Achitouv 2024
# this code take a input the training and testing datasets for the embedings of tweets given a CDA class and returns the NLPCA prediction from the test set.
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

from funccatMLauto import *
import os


dfank=read_emb_cat('/home/ixandra/Pclus/data/dfankers')
dfav=read_emb_cat('/home/ixandra/Pclus/data/dfusers')
ranuser=dfav['user_id'].unique().tolist()
ankers=dfank['user_id'].unique().tolist()
useralgo=ranuser+ankers
print('len anchors',len(ankers))
print('len dfank=',len(dfank))



cases_algo=['Modularity Class_0.005','Modularity Class_0.01','Modularity Class_0.02','Modularity Class_0.025','Modularity Class_0.05','Modularity Class_0.1','Modularity Class_0.5','Modularity Class_1','Modularity Class_05','Modularity Class_8','Modularity Class_10','Modularity Class_15','Modularity Class_20','Modularity Class_35']
filealgo='/home/ixandra/data/Gephi_clim29k.csv'
labelcol='Id'

print('**'*50,'start selecting nb cat')
Ncatcase=[]
for case in cases_algo:
    dftemp=pd.read_csv(filealgo)
    dftemp=dftemp[dftemp[case]!=-1] # remove border users
    Ncat=len(set(dftemp[case].tolist()))
    print('Ncat=',Ncat)
    dicalgo_user,frac_cat=makedicalgo(case,Ncat,useralgo,filealgo,labelcol)
    #print(frac_cat)
    print('training set')
    Nmax=calc_fractweets_percat(dfank,dicalgo_user)
    if Nmax>100:
        Nmax=5
    Ncatcase.append(Nmax-1) # -1 ???????
    print('Nmax=',Nmax)
print('Ncat per case=',Ncatcase)

for i in range(len(Ncatcase)):
    nmax=max(3,Ncatcase[i])
    Ncatcase[i]=5 #nmax


print('**'*50,'start ML')


Npercat=25000 # training cut for balanced data 
Npercattest=25000 # test cut for balanced data 
Score1=[]
Score2=[]
Score3=[]
ok=1
for case,Ncat in zip(cases_algo,Ncatcase):
    print('**'*50)
    print('case=',case,'Ncat_case=',Ncatcase)
    newpath = r'/home/ixandra/Pclus/data/MLcat/case_'+case+'_Ncat='+str(Ncat)+'/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    dicalgo_user,frac_cat=makedicalgo(case,Ncat,useralgo,filealgo,labelcol)
    print(frac_cat)
    print('training set')
    calc_fractweets_percat(dfank,dicalgo_user)

    print('testset')
    calc_fractweets_percat(dfav,dicalgo_user)


    if ok==1:


        print('**'*10,'training')
        Xtrain,ytrain,userseltrain=makeXy(frac_cat,dicalgo_user,dfank,Npercat)

        print('**'*10,'testing')
        Xtest,ytest,userseltest=makeXy(frac_cat,dicalgo_user,dfav,Npercattest)

        print('**'*10,'class frac')
        print(frac_cat)

        wait = 1. #input("Press 1 to train on new dataset, 0  otherwise")

        models=read_or_writeML(int(wait),Xtrain,ytrain,newpath)

        stats_MLmodels(models,Xtest,ytest)

        models_sel=[models[0],models[1],models[3],models[5]]
        W=[1,3,1,2]
        ypred=average_predictions(models_sel,W,Xtest)

        ic=0
        icnor=0
        histw=[]
        for i in range(len(ytest)):
            if ytest[i]==ypred[i]:
                cat=ytest[i]
                icnor+=1#-ytest.count(cat)/len(ytest) #1./Ncat
                ic+=1
            else:
                histw.append(ytest[i])
        for cat in set(ytest):
            print('cat=',cat,'nbwrong=',histw.count(cat),'/',ytest.count(cat),'=',histw.count(cat)/ytest.count(cat))
        print('ensemble model on L','nb correct=',ic,'/',len(ytest),'frac of',ic/len(ytest))
        Ncatt=len(set(ytest))
        RD=len(ytest)/Ncatt
        Score1.append((icnor-RD)/(len(ytest)-RD))
        print('rescaled score',(icnor-RD)/(len(ytest)-RD))

        # write dict pred per cat for algo and NLP
        dic_userNLP,dic_useralgo,dic_err,Nvalcat,dic_err2 =average_user_cat(userseltest, dicalgo_user,ypred,ytest)
        dfNLP=pd.DataFrame(
        {'user_id': dic_userNLP.keys(),
         'catNLP': dic_userNLP.values(),
         'nb tweets': dic_err.values(),
         'nb cats': dic_err2.values()
        })
        dfalgo=pd.DataFrame(
        {'user_id': dic_useralgo.keys(),
         'catalgo': dic_useralgo.values()
        })
        dfalgo = dfalgo.merge(dfNLP, on='user_id', how='inner')


        dfalgo.to_csv(newpath+'df_catusertest.csv',index=False)

        Fl=[]
        for cat in set(dic_userNLP.values()):
            Nincat=list(dic_userNLP.values()).count(cat)
            Nalgoincat=list(dic_useralgo.values()).count(cat)
            R=Nvalcat[cat-1]/Nincat
            P=Nvalcat[cat-1]/Nalgoincat
            print(cat,'Recall=',R,'Precision=',P,'Fscore=',2*R*P/(R+P))
            if cat <Ncat and (R+P)!=0:
                Fl.append(2*R*P/(R+P))
        Score2.append(np.mean(Fl))
        print('Score2=',np.mean(Fl))

        icnor=0
        for user in dic_userNLP.keys():
            if dic_userNLP[user]==dic_useralgo[user]:
                val=list(dic_useralgo.values())
                cat=dic_useralgo[user]
                icnor+=1#-val.count(cat)/len(val)

        Score3.append(icnor/len(dic_userNLP.keys()))
        RD=len(dic_userNLP.keys())/len(set(dic_userNLP.values()))
        s3nor=(icnor-RD)/(len(dic_userNLP.keys())-RD)
        print('Score3',icnor/len(dic_userNLP.keys()),'rescaled score=',s3nor)




