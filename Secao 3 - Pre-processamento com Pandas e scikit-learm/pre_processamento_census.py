# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:10:49 2020

@author: Luana Barros
"""

import pandas as pd

base = pd.read_csv("census.csv")

'''
    1. Transformação de Varivaeis Categóricas
'''

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,-1].values

'''
    Vamos transformais algumas variaveis nominais em discretas, 
    para que possam ser utilizadas pelos algoritmos de ML
    
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_previsores = LabelEncoder()
from sklearn.compose import ColumnTransformer

# i representa a coluna que terá o valor transformado para discreto
for i in [1,3,5,6,7,8,9,13]:
    previsores[:,i] = labelencoder_previsores.fit_transform(previsores[:,i])


'''
    No entanto, considerar varivaveis nominais (ex sexo, raça) como numéricas
    faz com que dentro dos algoritmos, valores maiores signifiquem maior importância,
    por isso, uma forma de transformar essas variaveis de forma a não ter ordem, é utilizar
    o OneHotEncoder, que cria para um atributo, várias colunas as classificando em 0 ou 1.
        Ex: raça era uma coluna com white, black, etc...
        agora a coluna raça é substituída por outras colunas, como 'white', 'black' etc
        onde cada registro pode ter 1 ou 0 pra essas colunas.
        
    Vamos fazer isso pra aquele conjunto de colunas.
'''

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# vamos também transformar o label da classe para discreto
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

'''
    2. Escanolamento de Atributos
'''

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)



