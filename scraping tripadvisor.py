# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:48:00 2019

@author: ja.moreno
"""

import requests
from parsel import Selector

resultados=[]
rev = 0
punt_parc = []
coment = []
pagina = 0
url = 'https://www.tripadvisor.es/Hotel_Review-g187528-d1841204-Reviews-Allon_Mediterrania_Hotel-Villajoyosa_Costa_Blanca_Province_of_Alicante_Valencian_Count.html#REVIEWS'

while pagina < 88:
    try: 
        pagina = rev/5+1
        print (f'Página {pagina:1.0f}')
        r = requests.get(url)
        sel = Selector(r.text)

        comentario = sel.css("q span::text").extract()
        puntuacion = sel.css("div.hotels-review-list-parts-RatingLine__bubbles--1oCI4").extract()
        

        for i in comentario:
            if i != '…':
                coment.append(i)
        for pos in range(len(puntuacion)):
            tx = puntuacion[pos][121:123]
            if tx == 'nQ':
                sc=5
            if tx == '7B':
                sc=4
            if tx == '1L':
                sc=2
            if tx == '17':
                sc=3
            if tx == '1N':
                sc=1
            punt_parc.append(sc)

        
        rev=rev+5
        url = 'https://www.tripadvisor.es/Hotel_Review-g187528-d1841204-Reviews-or'+str(rev)+'-Allon_Mediterrania_Hotel-Villajoyosa_Costa_Blanca_Province_of_Alicante_Valencian_Count.html#REVIEWS'

    except:
        break
    
from textblob import TextBlob

sent = []
value = 0

for line in coment:
    analysis = TextBlob(line)
    eng=analysis.translate(to='en')
    sent.append(eng.sentiment.polarity)
    value += 1
    print(f"Procesado comentario {value:1.0f} de {len(coment):1.0f}")
    


SentRating = []
for i in range(len(sent)):
    if sent[i] < -0.3:
        SentRating.append(1)
    elif sent[i] < -0.1:
        SentRating.append(2)
    elif sent[i] < 0.2:
        SentRating.append(3)
    elif sent[i] < 0.4:
        SentRating.append(4)
    else:
        SentRating.append(5)
        
SentFinal=[]
for i in range(len(sent)):
    if sent[i] < -0.1:
        SentFinal.append(0)
    elif sent[i] < 0.2:
        SentFinal.append(1)
    else:
        SentFinal.append(2)
        
listafin = []
for t in range(327):
    elem = [coment[t], punt_parc[t], sent[t], SentRating[t], SentFinal[t]]
    listafin.append(elem)    
    
import pandas as pd


df = pd.DataFrame(listafin, columns=['Comentario', 'Rating', 'Sentimiento', 'SentRating', 'SentFinal'])


print (f"Media de Rating Usuarios: {df.Rating.mean():2.2f}")
print (f"Media de Sentimiento TextBlob: {df.SentRating.mean():2.2f}")


acierto = 0
filas = df.shape[0]
for t in range(filas):
    a = df.iloc[t][1]
    b = df.iloc[t][4]
    if a in (1,2) and b == 0:
        acierto = acierto + 1
    if a == 3 and b == 1:
        acierto = acierto + 1
    if a in (4,5) and b == 2:
        acierto = acierto + 1
        
acc_Sent = acierto/filas*100
res100 = [acc_Sent,'Sentimiento TextBlob']
resultados.append(res100)

# =============================================================================
# CON VECTORES
# =============================================================================

import requests
from parsel import Selector


rev = 0
punt_parc = []
coment = []
pagina = 0
url = 'https://www.tripadvisor.es/Hotel_Review-g187525-d264047-Reviews-Palm_Beach_Hotel-Benidorm_Costa_Blanca_Province_of_Alicante_Valencian_Country.html#REVIEWS'

while pagina < 298:
    try: 
        pagina = rev/5+1
        print (f'Página {pagina:1.0f}')
        r = requests.get(url)
        sel = Selector(r.text)

        comentario = sel.css("q span::text").extract()
        puntuacion = sel.css("div.hotels-review-list-parts-RatingLine__bubbles--1oCI4").extract()
        

        for i in comentario:
            if i != '…':
                coment.append(i)
        for pos in range(len(puntuacion)):
            tx = puntuacion[pos][121:123]
            if tx == 'nQ':
                sc=5
            if tx == '7B':
                sc=4
            if tx == '1L':
                sc=2
            if tx == '17':
                sc=3
            if tx == '1N':
                sc=1
            punt_parc.append(sc)

        
        rev=rev+5
        url = 'https://www.tripadvisor.es/Hotel_Review-g187525-d264047-Reviews-or'+str(rev)+'-Palm_Beach_Hotel-Benidorm_Costa_Blanca_Province_of_Alicante_Valencian_Country.html#REVIEWS'

    except:
        break

coment.pop(5)

import spacy
import es_core_news_sm
nlp = es_core_news_sm.load()
import pandas as pd

def normaliza(texto):
    doc = nlp(texto)
    tokens = [t.orth_ for t in doc] 
    lexical_tokens = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    words = [t.lower() for t in lexical_tokens if len(t) > 3 and t.isalpha()]
    return words

normalizados = []
for t in range(len(coment)):
    words=normaliza(str(coment[t]))
    print ("Comentario ",t, " de ",len(coment))
    normalizados.append(words)
    
lematizados = []
for t in range(len(normalizados)):
    print ("Comentario ",t, " de ",len(normalizados))
    frase = ''
    for u in range(len(normalizados[t])):
        frase = frase + normalizados[t][u]+' '
    doc = nlp(frase)
    lemmas = ([tok.lemma_.lower() for tok in doc])
    lematizados.append(lemmas)

tablafin=[]
for t in range(len(lematizados)):
    frase = ''
    for u in range(len(lematizados[t])):
        frase = frase + (lematizados[t][u]) + ' '
    tablafin.append(frase)
    
from sklearn import feature_extraction

# crear la transformación
vectorizer = feature_extraction.text.TfidfVectorizer(tablafin)

# tokenizar y construir vocabulario
X = vectorizer.fit_transform(tablafin)

Xd = X.todense()

# documento codificado
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

from sklearn.model_selection import train_test_split

y = []
for i in range(len(punt_parc)):
    if punt_parc[i] in (1,2):
        y.append(0)
    elif punt_parc[i] == 3:
        y.append(1)
    else:
        y.append(2)

X_train, X_test, y_train, y_test = train_test_split(Xd, y, test_size = 0.2)


# =============================================================================
# KNN
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

k = len(set(y_train))

clasificador_knn = KNeighborsClassifier(n_neighbors = k, weights = "uniform")
clasificador_knn.fit(X_train, y_train)
pred = clasificador_knn.predict(X_test)

acc = 0
for t in range(len(y_test)):
    if pred[t] == y_test[t]:
        acc = acc+1

acc_KNN=acc/len(y_test)*100

res100 = [acc_KNN,'KNN']
resultados.append(res100)
# =============================================================================
# RED NEURONAL ESTÁNDAR
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import numpy as np

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print(X_train.shape)

modelo = Sequential()
modelo.add(Dense(50, activation="relu", input_shape=(5261,)))
modelo.add(Dense(250, activation="relu"))
modelo.add(Dense(3, activation="softmax"))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train_one_hot, epochs = 100, batch_size = 1000, verbose = 2)

pred3 = modelo.predict(X_test)

pred_red3 = []

for t in range(len(pred3)):
    pred_red3.append(np.argmax(pred3[t]))

acc = 0
for t in range(len(y_test)):
    if pred_red3[t] == y_test[t]:
        acc = acc+1

acc_RN=acc/len(y_test)*100

res101 = [acc_RN,'RN Estándar']
resultados.append(res101)
# =============================================================================
# RED NEURONAL ESTÁNDAR CON REGULARIZACIÓN L2
# =============================================================================
from keras import regularizers

modelo = Sequential()
modelo.add(Dense(50, activation="relu", input_shape=(5261,)))
modelo.add(Dense(250, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
modelo.add(Dense(3, activation="softmax"))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train_one_hot, epochs = 100, batch_size = 1000, verbose = 2)

pred4 = modelo.predict(X_test)

pred_red4 = []

for t in range(len(pred4)):
    pred_red4.append(np.argmax(pred4[t]))

acc = 0
for t in range(len(y_test)):
    if pred_red4[t] == y_test[t]:
        acc = acc+1

acc_RN_L2=acc/len(y_test)*100

res104 = [acc_RN_L2,'RN L2']
resultados.append(res104)
    
# =============================================================================
# RED NEURONAL ESTÁNDAR CON REGULARIZACIÓN L1
# =============================================================================
from keras import regularizers

modelo = Sequential()
modelo.add(Dense(50, activation="relu", input_shape=(5261,)))
modelo.add(Dense(250, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
modelo.add(Dense(3, activation="softmax"))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train_one_hot, epochs = 100, batch_size = 1000, verbose = 2)

pred5 = modelo.predict(X_test)

pred_red5 = []

for t in range(len(pred5)):
    pred_red5.append(np.argmax(pred5[t]))

acc = 0
for t in range(len(y_test)):
    if pred_red5[t] == y_test[t]:
        acc = acc+1

acc_RN_L1=acc/len(y_test)*100

res105 = [acc_RN_L1,'RN L1']
resultados.append(res105)

# =============================================================================
# RED NEURONAL ESTÁNDAR CON DROPOUT
# =============================================================================
from keras.layers import Dropout


modelo = Sequential()
modelo.add(Dense(50, activation="relu", input_shape=(5261,)))
modelo.add(Dense(250, activation="relu"))
modelo.add(Dropout(0.2))
modelo.add(Dense(3, activation="softmax"))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train_one_hot, epochs = 100, batch_size = 1000, verbose = 2)

pred6 = modelo.predict(X_test)

pred_red6 = []

for t in range(len(pred6)):
    pred_red6.append(np.argmax(pred6[t]))

acc = 0
for t in range(len(y_test)):
    if pred_red6[t] == y_test[t]:
        acc = acc+1

acc_DR=acc/len(y_test)*100

res106 = [acc_DR,'RN Dropout 20%']
resultados.append(res106)

# =============================================================================
# RED NEURONAL ESTÁNDAR CON BATCH NORMALIZATION
# =============================================================================
from keras.layers import BatchNormalization


modelo = Sequential()
modelo.add(Dense(50, activation="relu", input_shape=(5261,)))
modelo.add(Dense(250, activation="relu"))
modelo.add(BatchNormalization())
modelo.add(Dense(3, activation="softmax"))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train_one_hot, epochs = 100, batch_size = 1000, verbose = 2)

pred7 = modelo.predict(X_test)

pred_red7 = []

for t in range(len(pred7)):
    pred_red7.append(np.argmax(pred7[t]))

acc = 0
for t in range(len(y_test)):
    if pred_red7[t] == y_test[t]:
        acc = acc+1

acc_BN=acc/len(y_test)*100

res107 = [acc_BN,'RN Batch Normalization']
resultados.append(res107)

# =============================================================================
# RED NEURONAL ESTÁNDAR CON BATCH NORMALIZATION + DROPOUT
# =============================================================================

modelo = Sequential()
modelo.add(Dense(50, activation="relu", input_shape=(5261,)))
modelo.add(Dense(250, activation="relu"))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.2))
modelo.add(Dense(3, activation="softmax"))
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.fit(X_train, y_train_one_hot, epochs = 100, batch_size = 1000, verbose = 2)

pred8 = modelo.predict(X_test)

pred_red8 = []

for t in range(len(pred8)):
    pred_red8.append(np.argmax(pred8[t]))

acc = 0
for t in range(len(y_test)):
    if pred_red8[t] == y_test[t]:
        acc = acc+1

acc_BNDO=acc/len(y_test)*100

res108 = [acc_BNDO,'RN Batch Normalization + DropOut']
resultados.append(res108)

# =============================================================================
# KMEANS
# =============================================================================

from sklearn.cluster import KMeans
estimador_kmedias = KMeans(random_state=42, n_clusters=3)
estimador_kmedias.fit(X_train)

pred9 = estimador_kmedias.predict(X_test)

acc = 0
for t in range(len(y_test)):
    if pred9[t] == y_test[t]:
        acc = acc+1

acc_KMeans=acc/len(y_test)*100

res109 = [acc_KMeans,'KMeans']
resultados.append(res109)

# =============================================================================
# SVM
# =============================================================================

from sklearn.svm import SVC
estimador_svm = SVC(kernel="linear")
estimador_svm.fit(X_train, y_train)
pred11 = estimador_svm.predict(X_test)

acc = 0
for t in range(len(y_test)):
    if pred11[t] == y_test[t]:
        acc = acc+1
        
acc_SVM_linear=acc/len(y_test)*100

res1011 = [acc_SVM_linear,'SVM(Kernel linear)']
resultados.append(res1011)

from sklearn.svm import SVC
estimador_svm = SVC(kernel="poly")
estimador_svm.fit(X_train, y_train)
pred12 = estimador_svm.predict(X_test)

acc = 0
for t in range(len(y_test)):
    if pred12[t] == y_test[t]:
        acc = acc+1
        
acc_SVM_poly=acc/len(y_test)*100

res1012 = [acc_SVM_poly,'SVM(Kernel poly)']
resultados.append(res1012)
# =============================================================================
# 
# from sklearn.svm import SVC
# estimador_svm = SVC(kernel="poly", degree = 6)
# estimador_svm.fit(X_train, y_train)
# pred13 = estimador_svm.predict(X_test)
# 
# acc = 0
# for t in range(len(y_test)):
#     if pred13[t] == y_test[t]:
#         acc = acc+1
#         
# acc_SVM_poly6=acc/len(y_test)*100
# 
# res1013 = [acc_SVM_poly6,'SVM(Kernel poly degree 6)']
# resultados.append(res1013)
# 
# from sklearn.svm import SVC
# estimador_svm = SVC(kernel="rbf")
# estimador_svm.fit(X_train, y_train)
# pred14 = estimador_svm.predict(X_test)
# 
# acc = 0
# for t in range(len(y_test)):
#     if pred14[t] == y_test[t]:
#         acc = acc+1
#         
# acc_SVM_rbf=acc/len(y_test)*100
# 
# res1014 = [acc_SVM_rbf,'SVM(Kernel rbf)']
# resultados.append(res1014)
# 
# from sklearn.svm import SVC
# estimador_svm = SVC(kernel="rbf", gamma = 100)
# estimador_svm.fit(X_train, y_train)
# pred15 = estimador_svm.predict(X_test)
# 
# acc = 0
# for t in range(len(y_test)):
#     if pred15[t] == y_test[t]:
#         acc = acc+1
#         
# acc_SVM_rbf100=acc/len(y_test)*100
# 
# res1015 = [acc_SVM_rbf100,'SVM(Kernel rbf gamma 100)']
# resultados.append(res1015)
# =============================================================================

# =============================================================================
# LISTA ORDENADA
# =============================================================================

resultados.sort()


import seaborn as sns
from seaborn import load_dataset
from matplotlib import pyplot




valores=[]
titulos=[]

for t in range(len(resultados)):
    valores.append(resultados[t][0])
    titulos.append(resultados[t][1])
    


df = pd.DataFrame(resultados, columns=['Accuracy', 'Método'])
pyplot.figure(figsize=(15, 10))
sns.barplot(y='Método',x='Accuracy', data=df, ci=0.4)





