# This code was writen by I. Achitouv 2024 see arXiv #####
# this is a code that run training on NLP classification and return the model or results on the testing set.
import numpy as np
from time import process_time
import pandas as pd
#import matplotlib.pyplot as plt
import pickle as pkl
import random
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('vinai/bertweet-base')
#from transformers import pipeline
#from transformers import AutoTokenizer
#from transformers import T5Tokenizer, T5ForConditionalGeneration
#tokenizert5 = T5Tokenizer.from_pretrained("t5-small")
#modelt5 = T5ForConditionalGeneration.from_pretrained("t5-small")

pd.options.mode.chained_assignment = None  # default='warn'
import re
from string import punctuation



######################### ML functions
def read_emb_cat(name):
    import pandas as pd
    dfall=pd.read_csv(name)
    dfall['emb'] = dfall['emb'].apply(lambda x:
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))

    dfall = dfall.dropna()

    #dfall=dfall.loc[dfall['user_id'].isin(userinter)]

    dfall = dfall.sample(frac = 1)
    dfall.sort_values(by='days', inplace=True)
    #dfall['cat']=dfall['user_id'].apply(lambda x: dic_[x])

    return dfall


######################### cat sel
def catselec(lcat,fracut):

    cats=list(set(lcat))
    frac=[]

    for cat in cats:
        #print('cat=',cat,'frac=',lcat.count(cat)/len(lcat))
        frac.append(lcat.count(cat)/len(lcat))

    ic=0
    #fracut=0.8
    catsel=[]
    catdic={}
    while ic<fracut:
        max_ind=frac.index(max(frac))
        ic+=max(frac)
        catsel.append(cats[max_ind])
        catdic[cats[max_ind]]=frac[max_ind]
        frac[max_ind]=-1


    return catsel,ic,catdic


def makedicalgo(case,Ncat,useralgo,filealgo,labelcol):

    dftemp=pd.read_csv(filealgo)

    #listcat=dftemp[case].tolist()
    #print('nbcat before removing bord',len(set(listcat)))


    dftemp=dftemp[dftemp[case]!=-1] # remove border users
    print('Ncat in=',Ncat)
    frac=0.999999999
    listcat=dftemp[case].tolist()
    print('nbcat after removing bord',len(set(listcat)),min(listcat),max(listcat))

    catsel,fracuser,catdic=catselec(listcat,frac)
    newcat={}
    ic=1
    for cat in catdic.keys():
        if ic <Ncat:
            newcat[cat]=ic
            ic+=1
        else:
            newcat[cat]=Ncat

    dftemp=dftemp.loc[dftemp[labelcol].isin(useralgo)]
    dic_={}

    for i in range(len(dftemp)):
        user=dftemp[labelcol].iloc[i]

        ctemp=dftemp[case].iloc[i]
        dic_[user]=newcat[ctemp]


    fraccat={}

    for cat in range(1,Ncat+1):
        fraccat[cat]=sum(value == cat for value in dic_.values())/len(dic_)


    return dic_,fraccat

def calc_fractweets_percat(df,dic_):

    cat_df=[]
    for i in range(len(df)):
        user=df['user_id'].iloc[i]
        if user in dic_.keys():
            cat_df.append(dic_[user])

    for cat in set(dic_.values()):
        print('cat', cat, '# tweets', cat_df.count(cat), 'frac=',cat_df.count(cat)/len(cat_df), 'frac_algo=', sum(value == cat for value in dic_.values())/len(dic_) )
        if cat_df.count(cat)<25000:  #5000:
            print('if training do not trust ML on cat',cat,'or reduce Ncat!!!')
            return cat
    pass



def makeXy(catdic_,dic_,df,Npercat):
    Ncat=len(catdic_)
    #Npercat=6000 #len(dftrain)/Ncat
    from sklearn.utils import shuffle
    X=[]
    y=[]
    usersel=[]
    for i in range(len(df)):
        user=df['user_id'].iloc[i]
        if user in dic_.keys():
            catuser=dic_[user]
            if catuser in catdic_.keys():
                emb=df['emb'].iloc[i]
                if y.count(catuser)<=Npercat:
                    y.append(catuser)
                    X.append(emb)
                    usersel.append(user)

    X,y,usersel=shuffle(X,y,usersel)
    for cat in catdic_.keys():
        print('cat=',cat,'nb tweets=',y.count(cat),'frac=',y.count(cat)/len(y))
    return X,y,usersel



def read_or_writeML(fitmodel,Xtrain,ytrain,name):
    import joblib
    from sklearn.linear_model import SGDClassifier
    model1 = SGDClassifier(loss="hinge", penalty="elasticnet", max_iter=500)

    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model2 = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

    from sklearn.naive_bayes import GaussianNB
    model3 = GaussianNB()


    # these models works better with rescaling:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    model4=make_pipeline(StandardScaler(),MLPClassifier(solver='lbfgs', alpha=0.01,learning_rate='adaptive',
                         hidden_layer_sizes=(5,2 ),max_iter=20000, warm_start=True ,random_state=10))

    from sklearn.neighbors import NearestCentroid
    model5 = make_pipeline(StandardScaler(),NearestCentroid())

    from sklearn.ensemble import RandomForestClassifier
    model6 = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=250))

    models = [model1, model2,model3, model4,model5,model6]
    #modelsr = [m1, m2,m3,m4,m5,m6]

    #fitmodel=0
    print('start:',name)

    if fitmodel==1:
        #print('start fitting models')
        ic=1
        print('uncomment .dump to save the models !!!')
        for model in models:
            #print('fiting',str(model),model)
            model.fit(Xtrain,ytrain)
            #joblib.dump(model, name+'/model'+str(ic)+'.joblib')
            ic+=1
    else:

       # print('read models')
        m1=joblib.load(name+'/model1.joblib')
        m2=joblib.load(name+'/model2.joblib')
        m3=joblib.load(name+'/model3.joblib')
        m4=joblib.load(name+'/model4.joblib')
        m5=joblib.load(name+'/model5.joblib')
        m6=joblib.load(name+'/model6.joblib')

        models=[m1,m2,m3,m4,m5,m6]
    return models
# Create a function to fit models and make predictions
def predict(model, Xtest):
    return model.predict(Xtest)

# Combine predictions and take the one with highest agreement btw models
def average_predictions(models,W, Xtest):
    predictions = []
    predictiontemp=[]
    ic=0
    for model in models:
        preds = predict(model,Xtest)
        predictiontemp.append(preds)
    #predictions=(np.array(predictions).reshape(len(Xtest),len(models)))
        predictions=predictiontemp*W[ic]
        ic+=1

    res=[]
    for i in range(len(Xtest)):
        l=[]
        #for j in range(len(models)):
        for j in range(sum(W)):

            l.append(predictions[j][i])

        res.append(max(set(l), key=l.count))

    return res


def stats_MLmodels(models,Xtest,ytest):

    for model in models:
        ypred=predict(model,Xtest)
        ic=0
        histw=[]
        for i in range(len(ytest)):
            if ytest[i]==ypred[i]:
                ic+=1
            else:
                histw.append(ytest[i])
        print('model=',model,'nb correct=',ic,'/',len(ytest),'frac of',ic/len(ytest))

        for cat in set(ytest):
            print('cat=',cat,'nbwrong=',histw.count(cat),'/',ytest.count(cat),'=',histw.count(cat)/ytest.count(cat))
    pass

def average_user_cat(userseltest_, dic_,ypred_,ytest_):
    users=set(userseltest_)
    print('there are',len(users),'users in the test set')
    dic_user={}
    dic_useralgo={}

    for user in users:
        dic_user[user]=[]
        dic_useralgo[user]=[]


    for i in range(len(ypred_)):
        user=userseltest_[i]

        dic_user[user].append(ypred_[i])
        dic_useralgo[user]=ytest_[i]

    validcat=[]
    dic_userNLP={}
    div=[]
    dic_err={}
    dic_err2={}
    Nvalcat=np.zeros(len(set(ypred_)))
    for user in users:
        lopinion=dic_user[user]
        av=(max(lopinion, key=lopinion.count))

        if av==dic_useralgo[user]:
            validcat.append(1)
            Nvalcat[av-1]+=1
        else:
            validcat.append(0)

        dic_err[user]=len((lopinion))
        dic_err2[user]=len(set(lopinion))  #/len(lopinion)
        dic_userNLP[user]=av
    print('valid cat=',sum(validcat),'/',len(users),'fracok=',sum(validcat)/len(users))

    return dic_userNLP,dic_useralgo,dic_err,Nvalcat,dic_err2

                        
