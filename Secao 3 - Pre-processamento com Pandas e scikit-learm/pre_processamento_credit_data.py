# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:30:44 2020

@author: Luana Barros
"""

import pandas as pd

base = pd.read_csv('credit_data.csv')
base.describe()

'''
    Tratamento de Valores Inconsistentes
'''
# vemos idades negativas
base.loc[base['age'] < 0] 
# é equivalente a
base[base.age < 0]

# Como resolver esse problema?

#1. apagar coluna (n vamos usar)
''' 
    base.drop('age', 1, inplace=True) 
    # o número 1 faz apagar a coluna inteira
    # o inplace=True faz com que o resultado não seja retornado para 
    # uma variavel, e sim modifique a própria variavel baseS
'''

#2. apagar somente os registros com problema
'''
    #apaga os indices (registros) que tem idade negativa
    base.drop(base[base.age < 0].index, inplace=True)
'''

#3. preencher os valores com a média das idades

# vamos calcular primeiro a média
base.mean()
base.age.mean()
base['age'].mean()
# note que esta média é inválida, pois foi calculada utilizando os valores negativos

# cálculo da média válida
media = base[base.age > 0].age.mean()
base.loc[base.age < 0, 'age'] = media

'''
    Tratamento de Valores Faltantes
'''
# vemos se existem idades com valores faltantes
base.loc[pd.isnull(base.age)]

# vamos dividir previsores e a classe em vars diferentes
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,-1].values

# utilizamos imputer para identificar valores por padrão NaN em toda a variavel previsores, 
# e preencher pela média por padrão
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(previsores)
previsores = imputer.transform(previsores)

'''
    Escalonamento de Atributos
'''

# temos a padronização (Standartisation) e a normalização (Normalization)
# Iremos utilizar a padronização, que é fazer uma mudança de escala para
# todos os valores x, utilizando a fórmula x = (x - mean(x))/std(x)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
