import pandas as pd

base = pd.read_csv('../datasets/risco_credito.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
                
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

for i in range(previsores.shape[1]):
    previsores[:,i] = labelencoder.fit_transform(previsores[:,i])

from sklearn.naive_bayes import GaussianNB
cll = GaussianNB()

# constroi a tabela de probabilidades
cll.fit(previsores, classe)

previsoes = cll.predict([[0,0,1,2],[3,0,0,0]])