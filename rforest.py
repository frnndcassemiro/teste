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

# scores = []
# avg_cv = []

# n_estimators = 10
rforest = RandomForestClassifier(n_estimators=10)
rforest = rforest.fit(x_train,y_train)
# scores.append(rforest.score(x_test, y_test))
print("10 -rforest.score(x_test, y_test):",rforest.score(x_test, y_test))
cv_scores10 = cross_val_score(rforest,x, y, cv=5)
print("10 - cv_scores:", cv_scores10)
# avg_cv.append(np.mean(cv_scores1))
print("10 - cv_avg:", np.mean(cv_scores10))

# n_estimators = 50
rforest = RandomForestClassifier(n_estimators=50)
rforest = rforest.fit(x_train,y_train)
# scores.append(rforest.score(x_test, y_test))
print("50 - rforest.score(x_test, y_test):",rforest.score(x_test, y_test))
cv_scores50 = cross_val_score(rforest,x, y, cv=5)
print("50 - cv_scores:", cv_scores50)
# avg_cv.append(np.mean(cv_scores50))
print("50 - cv_avg:", np.mean(cv_scores50))

# n_estimators = 100
rforest = RandomForestClassifier(n_estimators=100)
rforest = rforest.fit(x_train,y_train)
# scores.append(rforest.score(x_test, y_test))
print("100 - rforest.score(x_test, y_test):",rforest.score(x_test, y_test))
cv_scores100 = cross_val_score(rforest,x, y, cv=5)
print("100 - cv_scores:", cv_scores100)
# avg_cv.append(np.mean(cv_scores100))
print("100 - cv_avg:", np.mean(cv_scores100))

# n_estimators = 200
rforest = RandomForestClassifier(n_estimators=200)
rforest = rforest.fit(x_train,y_train)
# scores.append(rforest.score(x_test, y_test))
print("200 - rforest.score(x_test, y_test):",rforest.score(x_test, y_test))
cv_scores200 = cross_val_score(rforest,x, y, cv=5)
print("200 - cv_scores:", cv_scores200)
# avg_cv.append(np.mean(cv_scores200))
print("200 - cv_avg:", np.mean(cv_scores200))

# gerar  gráfico curvas roc
y_pred_proba = rforest.predict_proba(x_test)
skplt.metrics.plot_roc_curve(y_test,y_pred_proba)
plt.show()

# avg_cv.append(0.3150072361095906) 
# avg_cv.append(0.3227799471143751) 
# avg_cv.append(0.3202220777172312)
# avg_cv.append(0.3199790972702137)
# print ('done!')

# scores.append(0.6379710005914984) 
# scores.append(0.6460921138874235) 
# scores.append(0.6476268124630313) 
# scores.append(0.6473870158105927)
# print ('done!')


# avg_cv.append(0.31174927748748277) 
# avg_cv.append(0.317555533115106) 
# avg_cv.append(0.3187992953889669)
# print ('done!')
# print ('avg[0]:',avg[0])
# print ('avg[1]:',avg[1])
# print ('avg[1]:',avg[2])

# scores.append(0.6406567230988122) 
# scores.append(0.6460121816699439) 
# scores.append(0.6470513004971784)
# print ('done!')
# print ('scores[0]:',scores[0])
# print ('scores[1]:',scores[1])
# print ('scores[2]:',scores[2])

# gerar gráfico de acurácia sem cross validation
# plt.plot([10,50,100],scores,color = 'blue')
# plt.xlabel('Número de árvores na floresta')
# plt.ylabel('Acurácia')
# plt.title('Acurácia/Número de árvores na floresta')
# plt.show()

# gerar gráfico de acurácia média por número de árvores na floresta
# plt.plot([10,50,100],avg_cv,color = 'blue')
# plt.xlabel('Número de árvores na floresta')
# plt.ylabel('Acurácia média com cross validation')
# plt.title('Acurácia média/Número de árvores com cross validation')
# plt.show()

