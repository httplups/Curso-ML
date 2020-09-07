import pandas as pd

base = pd.read_csv('../datasets/census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

'''from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# aqui escalonamos apenas varivaveis que não são dummy, pois elas escalonadas
# pioaram o algoritmo (acc = 47%)

previsores[:,102:] = scaler.fit_transform(previsores[:,102:])

fazer scaler sem onhotencoder fica melhor (acc 80%)
'''

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB
cll = GaussianNB()

# constroi a tabela de probabilidades
cll.fit(previsores_treinamento, classe_treinamento)

previsores = cll.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao =  accuracy_score(classe_teste, previsores)
matriz = confusion_matrix(classe_teste, previsores)