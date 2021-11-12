'''Cargue las bibliotecas'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from sklearn import tree
import pydotplus
from sklearn.preprocessing import LabelEncoder

#cargamos el archivo con el cual trabajaremos
dataset = pd.read_csv('data.csv')
print(dataset.head(5))
dataset.info()
dataset.describe()
dataset.columns = ['Age', 'Gender','polyuria', 'Polydipsia', 'sudden', 'weakness', 'Polyphagia', 'Genital', 'visual',
                 'Itching', 'Irritability', 'delayed', 'partial', 'muscle', 'Alopecia', 'Obesity', 'clase']
print(dataset.columns)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.Gender = le.fit_transform(dataset.Gender)
dataset.polyuria = le.fit_transform(dataset.polyuria)
dataset.Polydipsia = le.fit_transform(dataset.Polydipsia)
dataset.sudden = le.fit_transform(dataset.sudden)
dataset.weakness = le.fit_transform(dataset.weakness)
dataset.Polyphagia = le.fit_transform(dataset.Polyphagia)
dataset.Genital = le.fit_transform(dataset.Genital)
dataset.visual = le.fit_transform(dataset.visual)
dataset.Itching = le.fit_transform(dataset.Itching)
dataset.Irritability = le.fit_transform(dataset.Irritability)
dataset.delayed = le.fit_transform(dataset.delayed)
dataset.partial = le.fit_transform(dataset.partial)
dataset.muscle = le.fit_transform(dataset.muscle)
dataset.Alopecia = le.fit_transform(dataset.Alopecia)
dataset.Obesity = le.fit_transform(dataset.Obesity)
dataset.clase = le.fit_transform(dataset.clase)
print(dataset)
print(dataset.describe())
print(dataset.info())

#forma visual las variables utilizando el tipo de gráfico boxplot
dataset.boxplot(column=['Age', 'clase'])
plt.show()
dataset.boxplot()
x = dataset.iloc[:,0:16].values
y = dataset.iloc[:, 16].values
#conjunto de datos en 75% (Entrenamiento) y 25% (Test) utilizando la función train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#entrenamiento de el arbol de decicion
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, random_state = 0)
print(classifier.fit(x_train, y_train))
y_pred = classifier.predict(x_test)
print(y_pred)
print(y_test)
#matriz de confucion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#accuracy
from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("F1 Score: ", metrics.f1_score(y_test, y_pred, average = 'weighted'))
print("ROC: ", metrics.roc_auc_score(y_test, y_pred))
print("Recall: ", metrics.recall_score(y_test, y_pred, average = 'weighted'))
from sklearn import tree
tree.export_graphviz(classifier, out_file = 'tree_social.dot')
dot_data = tree.export_graphviz(classifier,
                                out_file = None,
                                class_names = ['negative', 'positive'],
                                feature_names = list(dataset.drop(['clase'], axis=1)),
                                filled= True
                               )
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("Arbol.png")





