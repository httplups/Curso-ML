import pandas as pd

base = pd.read_csv('../datasets/census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_previsores = LabelEncoder()
for i in range(previsores.shape[1]):
    previsores[:,i] = labelencoder_previsores.fit_transform(previsores[:,i])


labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

'''
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()
'''

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.tree import DecisionTreeClassifier
cll = DecisionTreeClassifier(criterion='entropy', random_state=0)
cll.fit(previsores_treinamento, classe_treinamento)

resultados = cll.predict(previsores_teste)

from sklearn.metrics import accuracy_score, confusion_matrix
precisao = accuracy_score(classe_teste, resultados)
matriz = confusion_matrix(classe_teste, resultados)