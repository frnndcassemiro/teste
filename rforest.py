from dataf import main
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import *

import scikitplot as skplt
import matplotlib.pyplot as plt  
import numpy as np

x, y = main()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

scores = []
avg_cv = []

# n_estimators = 10
rforest = RandomForestClassifier(n_estimators=10)
rforest = rforest.fit(x_train,y_train)
scores.append(rforest.score(x_test, y_test))
print("Score para 10 árvores:",scores[0])
cv_scores1 = cross_val_score(rforest,x, y, cv=5)
# print("cv:", cv_scores1)
avg_cv.append(np.mean(cv_scores1))
print("avg_cv[0]:",avg_cv[0])

# n_estimators = 50
rforest = RandomForestClassifier(n_estimators=50)
rforest = rforest.fit(x_train,y_train)
scores.append(rforest.score(x_test, y_test))
print("Score para 50 árvores:",scores[1])
cv_scores2 = cross_val_score(rforest,x, y, cv=5)
# print("cv:", cv_scores2)
avg_cv.append(np.mean(cv_scores2))
print("avg_cv[1]:",avg_cv[1])

# n_estimators = 100
rforest = RandomForestClassifier(n_estimators=100)
rforest = rforest.fit(x_train,y_train)
scores.append(rforest.score(x_test, y_test))
print("Score para 100 árvores:",scores[2])
cv_scores3 = cross_val_score(rforest,x, y, cv=5)
# print("cv:", cv_scores3)
avg_cv.append(np.mean(cv_scores3))
print("avg_cv[2]:",avg_cv[2])

# n_estimators = 200
rforest = RandomForestClassifier(n_estimators=200)
rforest = rforest.fit(x_train,y_train)
scores.append(rforest.score(x_test, y_test))
print("Score para 200 árvores:",scores[3])
cv_scores4 = cross_val_score(rforest,x, y, cv=5)
# print("cv:", cv_scores4)
avg_cv.append(np.mean(cv_scores4))
print("avg_cv[3]:",avg_cv[3])

# gerar gráfico de acurácia sem cross validation
plt.plot([10,50,100,200],scores,color = 'blue')
plt.xlabel('Número de árvores na floresta')
plt.ylabel('Acurácia')
plt.title('Acurácia/Número de árvores na floresta')
plt.show()

# gerar gráfico de acurácia média por número de árvores na floresta
plt.plot([10,50,100,200],avg_cv,color = 'blue')
plt.xlabel('Número de árvores na floresta')
plt.ylabel('Acurácia média com cross validation')
plt.title('Acurácia média/Número de árvores com cross validation')
plt.show()

# gerar  gráfico curvas roc
y_pred_proba = rforest.predict_proba(x_test)
skplt.metrics.plot_roc_curve(y_test,y_pred_proba)
plt.show()