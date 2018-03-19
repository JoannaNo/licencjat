import numpy as np
import matplotlib.pyplot as py

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.cross_validation import KFold


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
gdzie1=np.where(y==1)[0] #wektor z indeksami obrazow zlosliwych
gdzie0=np.where(y==0)[0] #wektor z indeksami obrazow lagodnych
np.random.seed(75) #chodzi o to, by za kazdym razem tak samo ustawic random generator
np.random.shuffle(gdzie1)
np.random.shuffle(gdzie0)

print ("tyle jedynek", gdzie1.size)
print ("tyle zer", gdzie0.size)

kf=KFold(gdzie1.size,n_folds=folds)

#glowna petla cross-walidacyjna
print ("Rozpoczynanie petli cross walidacyjnej")

bestiF1=0
bestF1=0
bestTPR = 100 
#plik=open('wyniki_2.txt','w')
for i in range(0,2):
    #inicjalizacja trafnosci, precyzji, recall i f1 score
    #acc=np.zeros(folds)
    #pres=np.zeros(folds)
    #rec=np.zeros(folds)
    #f1=np.zeros(folds)
    b = np.arange(0,0.25,0.05)
    n = np.arange(0.25,0.9,0.03)
    m = np.arange(0.9,1.05,0.05)
    p = np.array(list(b)+list(n)+list(m)) #kolejne progi                       
    tpr = np.zeros(shape = (folds,len(p)))                                     
    fpr = np.zeros(shape = (folds,len(p)))                                     
    counter=0
    k = 0
    for train_index,test_index in kf:

        #obrazy testowe:
        true_test=np.append(gdzie1[test_index], gdzie0[test_index])
        y_test=y[true_test].astype(int)
        X_test=X_scaled[true_test]

        #obrazy treningowe:
        true_train=np.append(gdzie1[train_index], gdzie0[train_index]) 
        #? Jakie obrazy dosypujemy?
        #true_train=np.append(true_train, gdzie0[gdzie1.size:gdzie1.size+int(0.2*train_index.size)])
        X_train=X_scaled[true_train]
        y_train=y[true_train]        
        
        #indeksy obrazow nie wzietych do testu:
        #notest=np.setdiff1d(np.arange(y.size),true_test) 
       
        ktora_iteracja = k
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
            #precyzja1 = precision_score(y_test,z)
            rob_tpr = tp/(tp+fn)
            print('conf matrix: ', tn, fp, fn, tp)
            print('prawdopodobienstwo: ', j)
            rob_fpr = fp/(fp+tn)
            tpr[k,u] = rob_tpr
            print(tpr[k,u], "\n")
            fpr[k, u] = rob_fpr
            '''
            if abs(0.95-rob_tpr)< abs(0.95-bestTPR):
                bestTPR = rob_tpr
                best_j = j'''
        k += 1
        print('NASTEPNY PODZiAL')   
    sr_tpr = np.mean(tpr,axis=0)                                               
    sr_fpr = np.mean(fpr,axis=0)                                               
    bestTPR=[x for x in sr_tpr if abs(0.95-x)==np.min(abs(0.95-sr_tpr))][0]    
    best_j=[x*0.05 for x,_ in enumerate(p) if \
            abs(0.95-sr_tpr[x])==np.min(abs(0.95-sr_tpr))][0]  
    pred1_pro = pro[:,1]
    pred_best_j = [0 if a<best_j else 1 for a in pred1_pro]                    
    
#    for w in range(len(p)):
#        sr_tpr.append(np.mean(tpr[:,w]))
#        sr_fpr.append(np.mean(fpr[:,w])) 
    py.plot(sr_fpr,sr_tpr)
    yerr = np.std(tpr,axis=0)
    py.fill_between(sr_fpr, sr_tpr-yerr, sr_tpr+yerr,facecolor='b',alpha=0.1)
    py.xlabel("FPR")
    py.ylabel('TPR')    
        
    '''
        FPR,TPR,th=roc_curve(y_test,pro[:,1]) 
        print(th)
        py.plot(FPR,TPR)
    
     '''   
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
    print ("Finalowa trafnosc SVMa: ", acc, chr(177))#, np.std(acc))
    print ("Finalowe F1 score SVMa: ", (f1), chr(177))#, np.std(f1))
    print ("Finalowe recall SVMa: ", (rec), chr(177))#, np.std(rec))
    print ("Finalowe precision SVMa: ", (pres), chr(177))#, np.std(pres))
    print ("---")
 #   plik.write('ktora iteracja: '+str(i*1)+'\n'+'best TPR: '+ str(bestTPR) \
#               +'\nbest j: ' +str(best_j)+'\n'+"Finalowa trafnosc SVMa: "+ \
#               str(acc)+ '\n'+ "Finalowe F1 score SVMa: "+str(f1)+ '\n'+ \
#               "Finalowe recall SVMa: "+ str(rec)+ '\n'+ \
#               "Finalowe precision SVMa: " + str(pres)+ '\n'+"-------\n")
#    if np.mean(f1)>bestF1:
#        bestiF1=i*1
#        bestF1=np.mean(f1)

#print ('rbf')
#print (bestiF1)
#print (bestF1)
print('best TPR: ', bestTPR)
print('best j: ', best_j)




#plik.close()
