# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:37:15 2021

@author: David Cermeño
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)





url = 'diabetes.csv'
data = pd.read_csv(url)

#Normalizando datos diabetes
data.Glucose.unique()
rangosGlucose = [-1, 0, 30, 70, 90, 120, 160, 190,205]
nombresGlucose = ['1','2','3','4','5','6','7','8']
data.Glucose = pd.cut(data.Glucose, rangosGlucose, labels = nombresGlucose)

data.Insulin.unique()
rangosInsulin = [-1, 0, 30, 70, 90, 130, 150, 190, 220, 240, 280, 320, 360, 380, 420, 460, 500, 550, 580, 620, 680, 740, 760, 790, 820, 850]
nombresInsulin = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
data.Insulin = pd.cut(data.Insulin, rangosInsulin, labels = nombresInsulin)

data.BMI.unique()
rangosBMI = [-1, 0, 20.9, 30.9,50.9, 70.9]
nombresBMI = ['1','2','3','4','5']
data.BMI = pd.cut(data.BMI, rangosBMI, labels = nombresBMI)

data.DiabetesPedigreeFunction.unique()
rangosDiabetesPedigreeFunction = [-1, 0, 0.20, 0.300, 0.500, 0.700, 0.900, 1.300, 1.600, 1.800, 2.200, 2.500]
nombresDiabetesPedigreeFunction = ['1','2','3','4','5','6','7','8','9','10','11']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rangosDiabetesPedigreeFunction, labels = nombresDiabetesPedigreeFunction)


x = np.array(data.drop(['Outcome'], 1))
y = np.array(data.Outcome) # 0 No, 1 Si
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


def metricas_entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred


def matriz_confusion_auc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr


def show_roc_curve_matrix(fpr, tpr, matriz_confusion):
    sns.heatmap(matriz_confusion)
    plt.show()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def show_metrics(str_model, AUC, acc_validation, acc_test, y_test, y_pred):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))
    print(f'AUC: {AUC} ')

#Punto 2

modelLogisticRegression = LogisticRegression()
modelLogisticRegression, acc_validationLogisticRegression, acc_testLogisticRegression, y_predLogisticRegression = metricas_entrenamiento(modelLogisticRegression, x_train, x_test, y_train, y_test)
matriz_confusionLogisticRegression, AUCLogisticRegression, fprLogisticRegression, tprLogisticRegression  = matriz_confusion_auc(modelLogisticRegression, x_test, y_test, y_predLogisticRegression)
show_metrics('Linear Regression', AUCLogisticRegression, acc_validationLogisticRegression, acc_testLogisticRegression, y_test, y_predLogisticRegression)
show_roc_curve_matrix(fprLogisticRegression, tprLogisticRegression, matriz_confusionLogisticRegression)

modelDecisionTreeClassifier = DecisionTreeClassifier()
modelDecisionTreeClassifier, acc_validationDecisionTreeClassifier, acc_testDecisionTreeClassifier, y_predDecisionTreeClassifier = metricas_entrenamiento(modelDecisionTreeClassifier, x_train, x_test, y_train, y_test)
matriz_confusionDecisionTreeClassifier, AUCDecisionTreeClassifier, fprDecisionTreeClassifier, tprDecisionTreeClassifier  = matriz_confusion_auc(modelDecisionTreeClassifier, x_test, y_test, y_predDecisionTreeClassifier)
show_metrics('Decision Tree',AUCDecisionTreeClassifier, acc_validationDecisionTreeClassifier, acc_testDecisionTreeClassifier, y_test, y_predDecisionTreeClassifier)
show_roc_curve_matrix(fprDecisionTreeClassifier, tprDecisionTreeClassifier, matriz_confusionDecisionTreeClassifier)

modelKNeighborsClassifier = KNeighborsClassifier()
modelKNeighborsClassifier, acc_validationKNeighborsClassifier, acc_testKNeighborsClassifier, y_predKNeighborsClassifier = metricas_entrenamiento(modelKNeighborsClassifier, x_train, x_test, y_train, y_test)
matriz_confusionKNeighborsClassifier, AUCKNeighborsClassifier, fprKNeighborsClassifier, tprKNeighborsClassifier  = matriz_confusion_auc(modelKNeighborsClassifier, x_test, y_test, y_predKNeighborsClassifier)
show_metrics('KNeighborns',AUCKNeighborsClassifier, acc_validationKNeighborsClassifier, acc_testKNeighborsClassifier, y_test, y_predKNeighborsClassifier)
show_roc_curve_matrix(fprKNeighborsClassifier, tprKNeighborsClassifier, matriz_confusionKNeighborsClassifier)

modelGaussianNB, acc_validationGaussianNB, acc_testGaussianNB, y_predGaussianNB = metricas_entrenamiento(GaussianNB(), x_train, x_test, y_train, y_test)
matriz_confusionGaussian, AUCGaussianNB, fprGaussian, tprGaussian = matriz_confusion_auc(modelGaussianNB, x_test, y_test, y_predGaussianNB)
show_metrics('GaussianNB', AUCGaussianNB, acc_validationGaussianNB, acc_testGaussianNB, y_test, y_predGaussianNB)
show_roc_curve_matrix(fprGaussian, tprGaussian, matriz_confusionGaussian)

modelMLPClassifier = MLPClassifier()
modelMLPClassifier, acc_validationMLPClassifier, acc_testMLPClassifier, y_predMLPClassifier = metricas_entrenamiento(modelMLPClassifier, x_train, x_test, y_train, y_test)
matriz_confusionMLPClassifier, AUCMLPClassifier, fprMLPClassifier, tprMLPClassifier  = matriz_confusion_auc(modelMLPClassifier, x_test, y_test, y_predMLPClassifier)
show_metrics('MLP Classifier',AUCMLPClassifier, acc_validationMLPClassifier, acc_testMLPClassifier, y_test, y_predMLPClassifier)
show_roc_curve_matrix(fprMLPClassifier, tprMLPClassifier, matriz_confusionMLPClassifier)

modelGradientBoostingClassifier = GradientBoostingClassifier()
modelGradientBoostingClassifier, acc_validationGradientBoostingClassifier, acc_testGradientBoostingClassifier, y_predGradientBoostingClassifier = metricas_entrenamiento(modelGradientBoostingClassifier, x_train, x_test, y_train, y_test)
matriz_confusionGradientBoostingClassifier, AUCGradientBoostingClassifier, fprGradientBoostingClassifier, tprGradientBoostingClassifier  = matriz_confusion_auc(modelGradientBoostingClassifier, x_test, y_test, y_predGradientBoostingClassifier)
show_metrics('GrandientBossting',AUCGradientBoostingClassifier, acc_validationGradientBoostingClassifier, acc_testGradientBoostingClassifier, y_test, y_predGradientBoostingClassifier)
show_roc_curve_matrix(fprGradientBoostingClassifier, tprGradientBoostingClassifier, matriz_confusionGradientBoostingClassifier)


#Punto 3
datosMetricas = {'Metric':['GrandientBossting','Linear Regression','Gaussian NB','MLP Classifier','KNeighborns','Decision Tree'],
                     
         'Training Accuracy':[round(accuracy_score(y_test, y_predGradientBoostingClassifier),3),round(accuracy_score(y_test, y_predLogisticRegression),3),round(accuracy_score(y_test, y_predGaussianNB),3),
                     round(accuracy_score(y_test, y_predMLPClassifier),3),round(accuracy_score(y_test, y_predKNeighborsClassifier),3),round(accuracy_score(y_test, y_predDecisionTreeClassifier),3)],
         
        'Validation Accuracy':[round(acc_validationGradientBoostingClassifier,3),round(acc_validationLogisticRegression,3),round(acc_validationGaussianNB,3),
                                round(acc_validationMLPClassifier,3),round(acc_validationKNeighborsClassifier,3),round(acc_validationDecisionTreeClassifier,3)],
                               
         'Test Accuracy':[round(acc_testGradientBoostingClassifier,3),round(acc_testLogisticRegression,3),round(acc_testGaussianNB,3),round(acc_testMLPClassifier,3),round(acc_testKNeighborsClassifier,3),round(acc_testDecisionTreeClassifier,3)],
         
         'Precision':[round(precision_score(y_test, y_predGradientBoostingClassifier),3),round(precision_score(y_test, y_predLogisticRegression),3),round(precision_score(y_test, y_predGaussianNB),3),
                     round(precision_score(y_test, y_predMLPClassifier),3),round(precision_score(y_test, y_predKNeighborsClassifier),3),round(precision_score(y_test, y_predDecisionTreeClassifier),3)],
                      
         'Recall':[round(recall_score(y_test, y_predGradientBoostingClassifier),3),round(recall_score(y_test, y_predLogisticRegression),3),round(recall_score(y_test, y_predGaussianNB),3),
                     round(recall_score(y_test, y_predMLPClassifier),3),round(recall_score(y_test, y_predKNeighborsClassifier),3),round(recall_score(y_test, y_predDecisionTreeClassifier),3)],
                   
         'F1-Score':[round(f1_score(y_test, y_predGradientBoostingClassifier),3),round(f1_score(y_test, y_predLogisticRegression),3),round(f1_score(y_test, y_predGaussianNB),3),
                     round(f1_score(y_test, y_predMLPClassifier),3),round(f1_score(y_test, y_predKNeighborsClassifier),3),round(f1_score(y_test, y_predDecisionTreeClassifier),3)],
                     
         'AUC':[round(AUCGradientBoostingClassifier,3),round(AUCLogisticRegression,3),round(AUCGaussianNB,3),round(AUCMLPClassifier,3),round(AUCKNeighborsClassifier,3),round(AUCDecisionTreeClassifier,3)]}

tablaMetricas = pd.DataFrame(datosMetricas).sort_values(by='AUC', ascending=False)
print(tablaMetricas)

#Punto 4

print ("Confusion matrix Logistic Regression:\n%s" % pd.crosstab(y_test, y_predLogisticRegression, rownames=['Actual'], colnames=['Predicted']))
print ("Confusion matrix Decision Tree:\n%s" % pd.crosstab(y_test, y_predDecisionTreeClassifier, rownames=['Actual'], colnames=['Predicted']))
print ("Confusion matrix KNeighborns:\n%s" % pd.crosstab(y_test, y_predKNeighborsClassifier, rownames=['Actual'], colnames=['Predicted']))
print ("Confusion matrix quadratic Gaussian NB:\n%s" % pd.crosstab(y_test, y_predGaussianNB, rownames=['Actual'], colnames=['Predicted']))
print ("Confusion matrix MLP Classifier:\n%s" % pd.crosstab(y_test, y_predMLPClassifier, rownames=['Actual'], colnames=['Predicted']))
print ("Confusion matrix GrandientBossting:\n%s" % pd.crosstab(y_test, y_predGradientBoostingClassifier, rownames=['Actual'], colnames=['Predicted']))


#Punto 5

fig = plt.figure(figsize = (15,15)) # width x height
ax1 = fig.add_subplot(3, 3, 1) # row, column, position
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)

sns.heatmap(matriz_confusionLogisticRegression, ax=ax1,annot=True,cbar_kws={'label': 'Linear Regression'})
sns.heatmap(matriz_confusionDecisionTreeClassifier, ax=ax2,annot=True,cbar_kws={'label': 'Decision Tree'})
sns.heatmap(matriz_confusionKNeighborsClassifier, ax=ax3, annot=True,cbar_kws={'label': 'KNeighborns'})
sns.heatmap(matriz_confusionGaussian, ax=ax4, annot=True,cbar_kws={'label': 'GaussianNB'})
sns.heatmap(matriz_confusionMLPClassifier, ax=ax5, annot=True,cbar_kws={'label': 'MLP Classifier'})
sns.heatmap(matriz_confusionGradientBoostingClassifier, ax=ax6, annot=True,cbar_kws={'label': 'GrandientBossting'})



#Punto 6
Metricas = {'Metric':['GrandientBossting','Linear Regression','Gaussian NB','MLP Classifier','KNeighborns','Decision Tree'],
         
         'Precision':[round(precision_score(y_test, y_predGradientBoostingClassifier),3),round(precision_score(y_test, y_predLogisticRegression),3),round(precision_score(y_test, y_predGaussianNB),3),
                     round(precision_score(y_test, y_predMLPClassifier),3),round(precision_score(y_test, y_predKNeighborsClassifier),3),round(precision_score(y_test, y_predDecisionTreeClassifier),3)],
                      
         'Recall':[round(recall_score(y_test, y_predGradientBoostingClassifier),3),round(recall_score(y_test, y_predLogisticRegression),3),round(recall_score(y_test, y_predGaussianNB),3),
                     round(recall_score(y_test, y_predMLPClassifier),3),round(recall_score(y_test, y_predKNeighborsClassifier),3),round(recall_score(y_test, y_predDecisionTreeClassifier),3)],
                   
         'F1-Score':[round(f1_score(y_test, y_predGradientBoostingClassifier),3),round(f1_score(y_test, y_predLogisticRegression),3),round(f1_score(y_test, y_predGaussianNB),3),
                     round(f1_score(y_test, y_predMLPClassifier),3),round(f1_score(y_test, y_predKNeighborsClassifier),3),round(f1_score(y_test, y_predDecisionTreeClassifier),3)]}

MetricasPuntoSeis = pd.DataFrame(Metricas)
print(MetricasPuntoSeis)


#Punto 7   
modelGaussianNB
probsGaussianNB = modelGaussianNB.predict_proba(x_test)
probsGaussianNB = probsGaussianNB[:, 1]
fprGaussian, tprGaussian, _ = roc_curve(y_test, probsGaussianNB)
plt.plot(fprGaussian, tprGaussian, color='#003f5c', label='ROC gaussianNB')       
    
probsLogistRegression = modelLogisticRegression.predict_proba(x_test)
probsLogistRegression = probsLogistRegression[:, 1]
fprLogisticRegression, tprLogisticRegression, _ = roc_curve(y_test, probsLogistRegression)
plt.plot(fprLogisticRegression, tprLogisticRegression, color='#444e86', label='ROC Linear Regression')
       
probsDecisionTreeClassifier = modelDecisionTreeClassifier.predict_proba(x_test)
probsDecisionTreeClassifier = probsDecisionTreeClassifier[:, 1]
fprDecisionTreeClassifier, tprDecisionTreeClassifier, _ = roc_curve(y_test, probsDecisionTreeClassifier)
plt.plot(fprDecisionTreeClassifier, tprDecisionTreeClassifier, color='#955196', label='ROC Decision Tree')
             
probsKNeighborsClassifier = modelKNeighborsClassifier.predict_proba(x_test)
probsKNeighborsClassifier = probsKNeighborsClassifier[:, 1]
fprKNeighborsClassifier, tprKNeighborsClassifier, _ = roc_curve(y_test, probsKNeighborsClassifier)
plt.plot(fprKNeighborsClassifier, tprKNeighborsClassifier, color='#dd5182', label='ROC KNeighborns')
      
probsMLPClassifier = modelMLPClassifier.predict_proba(x_test)
probsMLPClassifier = probsMLPClassifier[:, 1]
fprLogistic, tprLogistic, _ = roc_curve(y_test, probsMLPClassifier)
plt.plot(fprLogistic, tprLogistic, color='#ff6e54', label='ROC MLP Classifier')
    
probsGradientBoostingClassifier = modelGradientBoostingClassifier.predict_proba(x_test)
probsGradientBoostingClassifier = probsGradientBoostingClassifier[:, 1]
fprGradientBoostingClassifier, tprGradientBoostingClassifier, _ = roc_curve(y_test, probsGradientBoostingClassifier)
plt.plot(fprGradientBoostingClassifier, tprGradientBoostingClassifier, color='#ffa600', label='ROC GrandientBossting')
             
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
