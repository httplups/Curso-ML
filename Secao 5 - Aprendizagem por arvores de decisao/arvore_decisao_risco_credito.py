import pandas as pd

base = pd.read_csv('../datasets/risco_credito.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
                
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

for i in range(previsores.shape[1]):
    previsores[:,i] = labelencoder.fit_transform(previsores[:,i])

from sklearn.tree import DecisionTreeClassifier, export
cll = DecisionTreeClassifier(criterion='entropy')

# constroi a tabela de probabilidades
cll.fit(previsores, classe)

# historia, divida, garantias, renda
print(cll.feature_importances_)
resultados = cll.predict([[0,0,1,2],[3,0,0,0]])

# Para visualizar a árvore de decisão
export.export_graphviz(cll,
                       out_file = 'arvore.dot',
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = cll.classes_,
                       filled = True,
                       leaves_parallel=True)