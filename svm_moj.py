# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 23:27:28 2018

@author: Asia
"""

import numpy as np
import matplotlib.pyplot as py

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold
import random
import pandas as pd


#wczytywanie danych wejsciowych i wyjsciowych

#katalog="/dmj/fizmed/jginter/CNN/Czerniak1/eksperyment1/"
katalog=""
X=np.load(katalog+"eksp1_vgg_cnn_s_fc6.npy")
y=np.load("etykiety_eksp1.npy")

print ("Wymiar wejscia: ",np.shape(X))
print ("Wymiar etykiet: ",np.shape(y))

#skalowanie danych wejsciowych: 
#! moja wersja dziala szybciej niz ta Krzysztofa

def sigmoid_scaler2(A):
    B = np.divide(np.ones(A.shape),(1.0 + np.exp((-1.0) * A)))
    return B

X_scaled = sigmoid_scaler2(X)

#cross validation

#?
folds=5 #na ile podzbiorow podzielic dane
gdzie1=np.where(y==1)[0] #wektor z indeksami obrazow zlosliwych (248)
gdzie0=np.where(y==0)[0] #wektor z indeksami obrazow lagodnych  (1031)
np.random.seed(75) #chodzi o to, by za kazdym razem tak samo ustawic random generator
np.random.shuffle(gdzie1)
np.random.shuffle(gdzie0)
los=y.copy()
np.array(np.random.shuffle(los))

print ("tyle jedynek", gdzie1.size)
print ("tyle zer", gdzie0.size)

kf=KFold(gdzie1.size,n_folds=folds)

#glowna petla cross-walidacyjna
print ("Rozpoczynanie petli cross walidacyjnej")

bestiF1=0
bestF1=0
def prog ():
    df = pd.DataFrame({'C':[], 'TPR':[],'uTPR':[], 'prog j':[], 'acc':[],\
                       'u_acc':[], 'F1':[], 'u_F1':[], 'prec':[], 'u_prec':[]})
    bestTPR = 100 
    #plik=open('wyniki_zrownowazony_prbalos.txt','w')
    for i in range(0,9):
        #inicjalizacja trafnosci, precyzji, recall i f1 score
        #acc=np.zeros(folds)
        #pres=np.zeros(folds)
        #rec=np.zeros(folds)
        #f1=np.zeros(folds)
        b = np.arange(0,0.25,0.01)
        n = np.arange(0.25,0.9,0.03)
        m = np.arange(0.9,1.05,0.05)
        p = np.array(list(b)+list(n)+list(m)) #kolejne progi                       
        tpr = np.zeros(shape = (folds,len(p)))                                     
        fpr = np.zeros(shape = (folds,len(p)))
        tpr_luck = np.zeros(shape = (folds,len(p)))                                     
        fpr_luck = np.zeros(shape = (folds,len(p))) 
        traf_acc = np.zeros(shape = (folds,len(p)))                                     
        F1 = np.zeros(shape = (folds,len(p)))
        #rec_tpr = np.zeros(shape = (folds,len(p)))                                     
        precision = np.zeros(shape = (folds,len(p)))     
        traf_acc_luck = np.zeros(shape = (folds,len(p)))                                     
        F1_luck = np.zeros(shape = (folds,len(p)))
        #rec_tpr_luck = np.zeros(shape = (folds,len(p)))                                     
        precision_luck = np.zeros(shape = (folds,len(p)))                                
        counter=0
        k = 0
        for train_index,test_index in kf: #dzieli indeksy #248 na n_folds czesci
    
            reszta_0 = [x for x in gdzie0 if (x not in train_index and x not in test_index)]    
            #obrazy testowe:
            #true_test=np.append(gdzie1[test_index], gdzie0[test_index]) #indeksy 0 i 1 (po 248*0.2)=98 użyte do testow
            true_test=np.append(gdzie1[test_index], np.append(gdzie0[test_index], np.random.choice(reszta_0,len(test_index)*3,False)))
            y_test=y[true_test].astype(int) #wybrane y wziete do testow
            X_test=X_scaled[true_test] #wybrane X przeskalowane wziete do testow
            #klasyfikator losowy
            random.seed(3)
            #y_luck = [1 if random.random()>= .5 else 0 for x in range(len(y_test))] #jeszcze zly klasyfikator
#            y_luck = np.random.choice([0,1],len(y_test))
            y_luck=[0 if x<(4*len(y_test)//5) else 1 for x in range(len(y_test))]
            np.random.shuffle(y_luck)
            #print(y_luck)
    
            #obrazy treningowe:
            true_train=np.append(gdzie1[train_index],gdzie0[train_index] )   
            #true_train=np.append(gdzie1[train_index], np.append(gdzie0[train_index], np.random.choice(reszta_0,len(train_index)*3,False))) #indeksy 0 i 1 (po 248*0.8) użyte do treningu
            #? Jakie obrazy dosypujemy?
            #true_train=np.append(true_train, gdzie0[gdzie1.size:gdzie1.size+int(0.2*train_index.size)])
            X_train=X_scaled[true_train]
            y_train=y[true_train]  
           
            
            
            #indeksy obrazow nie wzietych do testu:
            #notest=np.setdiff1d(np.arange(y.size),true_test) 
           
#            ktora_iteracja = k                                                
            print('ktora iteracja: ', k)
            #! Wartosc C warto zwiekszac logarytmicznie
            C1=10**(float(i)/3)
            svm=SVC(kernel='rbf', C=C1, probability=True)
            svm.fit(X_train,list(y_train))
            params=svm.get_params()
            
            pred=svm.predict(X_test) #98 "0" lub '1'
            
            
    #       pred2=svm.predict(X_scaled)
            pro=svm.predict_proba(X_test) #lista list[prawd '0', prawd '1']
          
            for u,j in enumerate (p):
                z = pro[:,1]
                z = [0 if a<j else 1 for a in z]
                tn, fp, fn, tp = confusion_matrix(y_test, z).ravel()
                tn_l, fp_l, fn_l, tp_l = confusion_matrix(y_luck, z).ravel() #wartosci dla klasyfikatora losowego
                
                
                rob_tpr = tp/(tp+fn)
                rob_tpr_luck = tp_l/(tp_l+fn_l)
                rob_fpr = fp/(fp+tn)
                rob_fpr_luck = fp_l/(fp_l+tn_l)
                rob_traf_acc = (tp+tn)/(tp+tn+fp+fn)
                rob_F1 = 2*tp/(2*tp+fp+fn)
                rob_precision = tp/(tp+fp) 
                rob_traf_acc_luck = (tp_l+tn_l)/(tp_l+tn_l+fp_l+fn_l)
                rob_F1_luck = 2*tp_l/(2*tp_l+fp_l+fn_l)
                rob_precision_luck = tp_l/(tp_l+fp_l)
                
                
                
                tpr[k,u] = rob_tpr
                tpr_luck[k,u] = rob_tpr_luck
                fpr[k, u] = rob_fpr
                fpr_luck[k, u] = rob_fpr_luck
                traf_acc[k,u] = rob_traf_acc
                F1[k,u] = rob_F1
                precision[k,u] = rob_precision
                traf_acc_luck[k,u] = rob_traf_acc_luck
                F1_luck[k,u] = rob_F1_luck
                precision_luck[k,u] = rob_precision_luck
                
                ''' 
                if abs(0.95-rob_tpr)< abs(0.95-bestTPR):
                    bestTPR = rob_tpr
                    best_j = j'''
            k += 1
            print('NASTEPNY PODZiAL')   
        sr_tpr = np.mean(tpr,axis=0)                                               
        sr_fpr = np.mean(fpr,axis=0)
        sr_tpr_luck = np.mean(tpr_luck,axis=0)                                               
        sr_fpr_luck = np.mean(fpr_luck,axis=0)     
        sr_traf_acc =  np.mean(traf_acc,axis=0)  
        sr_F1 =  np.mean(F1,axis=0)
        sr_precision =  np.mean(precision,axis=0)  
        sr_traf_acc_luck =  np.mean(traf_acc_luck,axis=0)  
        sr_F1_luck =  np.mean(F1_luck,axis=0)
        sr_precision_luck =  np.mean(precision_luck,axis=0)   
        #best TPR
#        bestTPR=[x for x in sr_tpr if abs(0.95-x)==np.min(abs(0.95-sr_tpr))][0] 
#        ind = list(sr_tpr).index(bestTPR)
#        best_j= p[ind]
#        bestTPR_luck=[x for x in sr_tpr_luck if abs(0.95-x)==np.min(abs(0.95-sr_tpr_luck))][0] 
#        ind_luck = list(sr_tpr_luck).index(bestTPR_luck)
#        best_j_luck= p[ind_luck]
        
        #best F1
#        bestF1 = max(sr_F1)
#        ind = list(sr_F1).index(bestF1)                        
#        best_j = p[ind] 
#        bestF1_luck = max(sr_F1_luck)
#        print('best_F1: ', bestF1)
#        ind_luck = list(sr_F1_luck).index(bestF1_luck)                        
#        best_j_luck = p[ind_luck] 
        
        #best accuracy
#        bestAcc = max(sr_traf_acc)
#        ind = list(sr_traf_acc).index(bestAcc)                        
#        best_j = p[ind] 
#        best_acc_luck = max(sr_traf_acc_luck)
#        print('best_Acc: ', bestAcc)
#        ind_luck = list(sr_traf_acc_luck).index(best_acc_luck)                        
#        best_j_luck = p[ind_luck] 
        
        #best_precision
        best_pres = max(sr_precision)
        ind = list(sr_precision).index(best_pres)                        
        best_j = p[ind] 
        best_precision_luck = max(sr_precision_luck)
        print('best_prec: ', best_pres)
        ind_luck = list(sr_precision_luck).index(best_precision_luck)                        
        best_j_luck = p[ind_luck] 
        
        
        
        
        
        
        pred1_pro = pro[:,1]
        pred_best_j = [0 if a<best_j else 1 for a in pred1_pro]     
        #pred_best_j_luck = [0 if a<best_j_luck else 1 for a in pred1_pro]               
       
        yerr1 = np.std(tpr,axis=0)
        yerr2 = np.std(tpr_luck,axis=0)
#        d = auc(sr_fpr, sr_tpr)
#        d_luck = auc(sr_fpr_luck, sr_tpr_luck)
#        u_d = 0
#        u_d_luck = 0
#        for b in range(len(sr_fpr)-1):
#            u_d += (.5*(sr_fpr[b+1]-sr_fpr[b])*(yerr1[b+1]**2+yerr1[b]**2)**.5)**2
#            u_d_luck += (.5*(sr_fpr_luck[b+1]-sr_fpr_luck[b])*(yerr2[b+1]**2+yerr2[b]**2)**.5)**2
#        u_d=u_d**.5
#        u_d_luck=u_d_luck**.5
#            
#            
#            
#            
#        py.plot(sr_fpr,sr_tpr,label = 'AUC = {:.3f} {} {:.3f}'.format(d,chr(177),u_d))
#        py.plot(sr_fpr_luck,sr_tpr_luck, label ='AUC_luck = {:.3f} {} {:.3f}'.format(d_luck,chr(177),u_d_luck))
#        
#        py.fill_between(sr_fpr, sr_tpr-yerr1, sr_tpr+yerr1,facecolor='b',alpha=0.1)
#        py.fill_between(sr_fpr_luck, sr_tpr_luck-yerr2, sr_tpr_luck+yerr2,facecolor='g',alpha=0.1)
#        
#        py.xlabel("FPR")
#        py.ylabel('TPR')  
#        
#        py.legend(loc=4)
#        
#        py.savefig('ROC_svm_zrowtren_niezrow_C_'+str('{:.3f}'.format(C1))+'.png', dpi = 400)
##           
        '''
            FPR,TPR,th=roc_curve(y_test,pro[:,1]) 
            print(th)
            py.plot(FPR,TPR)
        
         '''  
        sr_traf_acc = sr_traf_acc[ind]  
        sr_F1 =  sr_F1[ind]
        sr_precision =  sr_precision[ind]
        sr_tpr = sr_tpr[ind]
         
        u_traf_acc =  np.std(traf_acc[:,ind])  
        u_F1 =  np.std(F1[:,ind])
        u_precision =  np.std(precision[:,ind])
        u_tpr = np.std(tpr[:,ind])
        
        sr_traf_acc_luck =  sr_traf_acc_luck[ind_luck]
        sr_F1_luck =  sr_F1_luck[ind_luck]
        sr_precision_luck =  sr_precision_luck[ind_luck]
        sr_tpr_luck = sr_tpr_luck[ind_luck]
        
        u_traf_acc_luck =  np.std(traf_acc_luck[:,ind_luck])  
        u_F1_luck =  np.std(F1_luck[:,ind_luck])
        u_precision_luck =  np.std(precision_luck[:,ind_luck])
        u_tpr_luck = np.std(tpr_luck[:,ind_luck])
        
        
        acc = accuracy_score(y_test,pred_best_j)
        pres = precision_score(y_test,pred_best_j)
        rec= recall_score(y_test,pred_best_j)
        f1 = f1_score(y_test,pred_best_j)
        counter += 1
            
        py.show()
        print (i*1, "C=", C1)
        print ('best TPR: ', bestTPR)
        print ('best j: ', best_j)
        print ("treningowych: ", X_train.shape, "testowych: ", X_test.shape)
        print ("treningowych jedynek:", np.sum(y_train))
        print ("Finalowa trafnosc SVMa: ", sr_traf_acc, "/",acc)#, np.std(acc))
        print ("Finalowe F1 score SVMa: ", sr_F1, '/', f1)#, np.std(f1))
        print ("Finalowe recall SVMa: ", sr_tpr, '/',rec)#, np.std(rec))
        print ("Finalowe precision SVMa: ", sr_precision, '/', pres)#, np.std(pres))
        print ("---")
        tdf=pd.DataFrame({'C':[C1], 'TPR':[sr_tpr], 'uTPR':[yerr1[ind]],'prog j':[best_j],\
                          'acc':[sr_traf_acc], 'u_acc':[u_traf_acc],'F1':[sr_F1],'u_F1':[u_F1],\
                          'prec':[sr_precision], 'u_prec':[u_precision]},index=[i])
        df=pd.concat([df,tdf])
        tdf=pd.DataFrame({'C':[C1], 'TPR':[sr_tpr_luck], 'uTPR':[yerr2[ind_luck]],'prog j':[best_j_luck],\
                          'acc':[sr_traf_acc_luck], 'u_acc':[u_traf_acc_luck],'F1':[sr_F1_luck],'u_F1':[u_F1_luck],\
                          'prec':[sr_precision_luck], 'u_prec':[u_precision_luck]},index=[i+15])
        df=pd.concat([df,tdf])
        
#        plik.write('ktora iteracja: '+str(i*1)+'\n'+'C: '+str(C1)+'\n'+'best TPR: '+ str(bestTPR) \
#                   +'\nbest j: ' +str(best_j)+'\n'+"Finalowa trafnosc SVMa: "+ \
#                   str(acc)+ '\n'+ "Finalowe F1 score SVMa: "+str(f1)+ '\n'+ \
#                   "Finalowe recall SVMa: "+ str(rec)+ '\n'+ \
#                   "Finalowe precision SVMa: " + str(pres)+ '\n'+"-------\n")
#        if np.mean(f1)>bestF1:
#            bestiF1=i*1
#            bestF1=np.mean(f1)
#    
    #print ('rbf')
    #print (bestiF1)
    #print (bestF1)
    print('best TPR: ', bestTPR)
    print('best j: ', best_j)
    df.to_excel('wyn_niezrow(tren)_zrow_svm_pres.xlsx',sheet_name='niezrow_zrown_svm_pres')
 #   plik.close()
    return 0

prog()
