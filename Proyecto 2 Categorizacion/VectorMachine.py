# librerias usadas para el proyecto
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
# Sklern usados para el proyecto
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
#Se lee el archivo .tsv que contiene los datos
tarjeta = pd.read_csv ('data/data.tsv',
                       header=1,
                       sep='\t')
#cambiar default payment next month a DEFAULT para trabajar más rápido, también se remueve ID que es inutiñ
tarjeta.rename({'default payment next month' : 'DEFAULT'}, axis='columns', inplace=True)
tarjeta.drop('ID', axis=1, inplace=True)
#Algunas lineas del .tsv tiene filas sin información, ya que tan solo son 64 de 30,000, es mejor borrarlas que repararlas ya que no afectara el programa e igual
#debmos aminorar la cantidad de datos
tarjeta_sin_perdida = tarjeta.loc[(tarjeta['EDUCATION'] !=0) & (tarjeta['MARRIAGE'] !=0)]
#Esta sección del codigo reduce las muestras ya que son más de 25,000 y son demasiadas para un SVM, se reducen a 2000 para manejar mejor el programa
tarjeta_sin_default = tarjeta_sin_perdida[tarjeta_sin_perdida['DEFAULT'] == 0]
tarjeta_default = tarjeta_sin_perdida[tarjeta_sin_perdida['DEFAULT'] == 1]
tarjeta_sin_default_reducido = resample(tarjeta_sin_default,
                                        replace=False,
                                        n_samples=1000,
                                        random_state=42)
tarjeta_default_reducido = resample(tarjeta_default,
                                        replace=False,
                                        n_samples=1000,
                                        random_state=42)

tarjeta_reducido = pd.concat([tarjeta_sin_default_reducido,tarjeta_default_reducido])
#Se le da formato a los datos mediante One Hot Encoding usando la ayuda de get_dummies

X = tarjeta_reducido.drop('DEFAULT', axis=1).copy()
y = tarjeta_reducido['DEFAULT'].copy()
X_Codificado = pd.get_dummies(X, columns=['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'])

#Centrando y escalando la los datos para probar y entrenar
X_train, X_test, y_train, y_test = train_test_split(X_Codificado, y, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
#Construyendo la maquina de vectores

'''
param_grid = [
    { 'C': [0.5, 1, 10, 100],
      'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
      'kernel': ['rbf']},
]

optimal_params = GridSearchCV (
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=0
)
optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)
'''
clf_svm = SVC (random_state=42, C = 100, gamma=0.001)
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix (clf_svm,
                       X_test_scaled,
                       y_test,
                       values_format='d',
                       display_labels=["No se fue a Default", "Se fue a Default"])
#Dibujando las graficas de comparación
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
plt.show()
per_var = np.round (pca.explained_variance_ratio_*100, decimals = 1)
labels = [str(x) for x in range(1, len(per_var)+1)]

plt.bar (x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(
    axis = 'x',
    which = 'both',
    bottom = False,
    top = False,
    labelbottom = False)

plt.ylabel('Percentage of Explained Variance')
plt.xlabel ('Principal Components')
plt.title('Screen Plot')
plt.show()
train_pc1_coords = X_train_pca[:,0]
train_pc2_coords = X_train_pca[:,1]
pca_train_scaled = scale(np.column_stack((train_pc1_coords,train_pc2_coords)))

clf_svm = SVC (random_state=42, C=1000, gamma=0.001)
clf_svm.fit(pca_train_scaled, y_train)
X_test_pca = pca.transform(X_train_scaled)
test_pc1_coords = X_test_pca[:,0]
test_pc2_coords = X_test_pca[:,1]
x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1
xx, yy = np.meshgrid (np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max,step=0.1))

Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(figsize = (10,10))
ax.contourf(xx, yy, Z, alpha = 0.1)
cmap = colors.ListedColormap(['#e41a1c','#4daf4a'])
scatter = ax.scatter (test_pc1_coords, test_pc2_coords,c=y_train,
                      cmap = cmap,
                      s=100,
                      edgecolors = 'k',
                      alpha =0.7)
legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc="upper right")
legend.get_texts()[0].set_text("No default")
legend.get_texts()[1].set_text("Yes default")
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title ('Decision surface using the PCA transformed/projected features')
plt.show()