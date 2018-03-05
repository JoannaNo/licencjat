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
bestTPR = 0 
for i in range(0,2):
    #inicjalizacja trafnosci, precyzji, recall i f1 score
    acc=np.zeros(folds)
    pres=np.zeros(folds)
    rec=np.zeros(folds)
    f1=np.zeros(folds)
    counter=0
    
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

        #! Wartosc C warto zwiekszac logarytmicznie
        C1=10**(float(i)/3)
        svm=SVC(kernel='rbf', C=C1, probability=True)
        svm.fit(X_train,list(y_train))
        params=svm.get_params()
        
        pred=svm.predict(X_test) #98 "0" lub '1'
        
        
#       pred2=svm.predict(X_scaled)
        pro=svm.predict_proba(X_test) #lista list[prawd '0', prawd '1']
      
        tpr = []
        fpr = []
        p = np.arange(0,1.05,0.05) #kolejne progi
        for i in p:
            z = pro[:,1]
            z = [0 if a<i else 1 for a in z]
            tn, fp, fn, tp = confusion_matrix(y_test, z).ravel()
            precyzja1 = precision_score(y_test,z)
            precyzja2 = tp/(tp+fn)
            print(precyzja1, precyzja2) # co liczy prec1 a co prec2?
            print('conf matrix: ', tn, fp, fn, tp)
            fp_fpr = fp/(fp+tn)
            tpr.append(precyzja2)
            fpr.append(fp_fpr)
        py.plot(fpr,tpr)    
            
        
        '''
        FPR,TPR,th=roc_curve(y_test,pro[:,1]) 
        print(th)
        py.plot(FPR,TPR)
        '''
        
        acc[counter] = accuracy_score(y_test,pred)
        pres[counter] = precision_score(y_test,pred)
        rec[counter] = recall_score(y_test,pred)
        f1[counter] = f1_score(y_test,pred)
        counter += 1
    py.show()
    print (i*1, "C=", C1)
    print ("treningowych: ", X_train.shape, "testowych: ", X_test.shape)
    print ("treningowych jedynek:", np.sum(y_train))
    print ("Finalowa trafnosc SVMa: ", np.mean(acc), "+/-", np.std(acc))
    print ("Finalowe F1 score SVMa: ", np.mean(f1), "+/-", np.std(f1))
    print ("Finalowe recall SVMa: ", np.mean(rec), "+/-", np.std(rec))
    print ("Finalowe precision SVMa: ", np.mean(pres), "+/-", np.std(pres))
    print ("---")
    if np.mean(f1)>bestF1:
        bestiF1=i*1
        bestF1=np.mean(f1)

print ('rbf')
print (bestiF1)
print (bestF1)
 #ii