#!/usr/bin/python
# -*- coding: ascii -*-
#import os, sys
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from scipy import stats
import xlwt

#skalowanie danych wejsciowych:

def sigmoid_scaler2(A):
    B = np.divide(np.ones(A.shape),(1.0 + np.exp((-1.0) * A)))
    return B

def testKolmogorowaSmirnowa(X,test_index,y,gdzie1,gdzie0):
    true_test=np.append(gdzie1[test_index], gdzie0[test_index])
    #y_test=y[true_test].astype(int)
    notest=np.setdiff1d(np.arange(y.size),true_test) #indeksy obrazow nie wzietych do testu
    nn=np.zeros((X.shape[1], 2))
    for jq in np.arange(X.shape[1]):
        sk, p = stats.ks_2samp(X[np.intersect1d(gdzie1,notest), jq], X[np.intersect1d(gdzie0, notest), jq])
        nn[jq, 0]=p
        nn[jq, 1]=jq
    nn=np.array(sorted(nn, key=lambda m: m[0], reverse=False))
    ind=nn[:,1].astype(int)
    #f.write( str(nn[0,0]*100)+"%"+","+ str(nn[ile, 0]*100)+"%"+ "\n")
    return ind

def testSrednich(X,test_index,y,gdzie1,gdzie0):
    true_test=np.append(gdzie1[test_index], gdzie0[test_index])
    #y_test=y[true_test].astype(int)
    notest=np.setdiff1d(np.arange(y.size),true_test) #indeksy obrazow nie wzietych do testu
    nn=[[iq,jq] for iq,jq in zip(np.abs(np.mean(X[np.intersect1d(gdzie1,notest)], axis=0)
            -np.mean(X[np.intersect1d(gdzie0, notest)], axis=0)), np.arange(X.shape[1]))]
    nn=np.array(sorted(nn, key=lambda m: m[0], reverse=True))
    ind=nn[:,1].astype(int)
    return ind


def petlaCrossWalidacji(D,y,folds,ile,test,ratio_test=1): 
    """D to lista krotek numery i ratio
    y tablica etykiet
    folds int ile folderow
    ile int ile parametrow
    test funkcja np.srednie i kolmogorow
    """
    tytul=["eksp1_vgg_cnn_s_fc6.npy"] 
    
    gdzie1=np.where(y==1)[0]
    gdzie0=np.where(y==0)[0]
    np.random.seed(12)
    np.random.shuffle(gdzie1)
    np.random.shuffle(gdzie0)
    kf=KFold(gdzie1.size,n_folds=folds)

    X=[]
    for i in D:
        X.append(np.load(tytul[i[0]]))

    indeksy=[]   
    #glowna petla cross-walidacyjna
    for k in range(1,8):
    
        for train_index,test_index in kf:
            #! wybor parametrow najlepiej roznicujacych dwie klasy
            ind=test(X[0],test_index,y,gdzie1,gdzie0)
            indeksy.append(ind)
        wspol=set(indeksy[0][0:int(4096/8*k)])
        for i in range(1, len(indeksy)):
            wspol=wspol & set(indeksy[i][0:int(4096/8*k)])
        sheet.write(k, 0, 4096/8*k)
        sheet.write(k, 1, len(wspol))

D=[(0,1)]
y=np.load("etykiety_eksp1.npy")
ile=2000
folds=5
book = xlwt.Workbook(encoding="utf-8")
sheet = book.add_sheet("KolmoSmir")
petlaCrossWalidacji(D,y,folds,ile,testKolmogorowaSmirnowa,ratio_test=1)
book.save("proba.xls")
