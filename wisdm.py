from __future__ import print_function
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import cPickle as pickle
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict 
import matplotlib.pyplot as plt
import pylab as pl


def sumLine(matriz,linha):
	k = 0
	i = linha
	for j in range (6):
		k += matriz[i][j]
	return k

def printData(datas):
	print()
	for w in range(len(datas['matriz'])):
		for z in range(len(datas['matriz'])):
			numero = datas['matriz'][w][z]
		
			if(numero >=100):
				print(" %d"%(numero),end ='')
			elif(numero >= 10):
				print("  %d"%(numero),end ='')
			else:
				print("   %d"%(numero),end ='')
		print()
	print()
	print("******Acurracies******")
	labels = datas['labels']
	x = 0
	y = 0
	for i in labels:
		div = float(sumLine(datas['matriz'],x))
		num = float(datas['matriz'][x][y])
		print("%s acurracy %.5f" %(i,datas['matriz'][x][y]/div))
		x+=1;
		y+=1;

	print()


def dataFrame(dataset, features, classes):
	data = pd.read_csv(dataset, na_values=['NaN'])
	x = data[features]
	y = data[classes]
	
	fillMean_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=0)
	fillMedian_NaN = Imputer(missing_values=np.nan, strategy='median', axis=0)
	fillFrequent_NaN = Imputer(missing_values=np.nan, strategy='most_frequent', axis=0)

	imputed_XMean = pd.DataFrame(fillMean_NaN.fit_transform(x))
	imputed_XMedian = pd.DataFrame(fillMedian_NaN.fit_transform(x))
	imputed_XFrequent = pd.DataFrame(fillFrequent_NaN.fit_transform(x))

	return {'features': x, 'featuresMean': imputed_XMean, 'featuresMedian': imputed_XMedian, 'featuresFrequent': imputed_XFrequent, 'classes': y}

def criar_folds(nfolds, serialFileName, dataset):
	# leitura do arquivo csv
	dF = dataset
	features = dF['features']
	classes = dF['classes']

	# criando os indices para os folds
	skf = StratifiedKFold(n_splits= nfolds, shuffle=True)

	# transforma numa lista e grava no arquivo 
	indices = list(skf.split(features, classes))

	# salvando em um arquivo
	pickle.dump(indices, open(serialFileName , "wb"))

def readFolds(serialFileName):
	return pickle.load(open(serialFileName, 'rb'))

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
def meanRemoval(dataframe):
	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaledX = scaler.fit_transform(dataframe)
	return preprocessing.scale(rescaledX)

def normalization(dataset):
	return preprocessing.normalize(dataset, norm="l2")

def fitandPredict(features,classes,serialFile,classifier ):
	matrix = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
	indices = readFolds(serialFile)
	for fold in range(len(indices)):
		trn_i = indices[fold][0]
		tst_i = set(np.array(range(0,len(classes)))) - set(np.array(trn_i))
		x_trn = features.ix[trn_i]
		y_trn = classes.ix[trn_i]
		classifier.fit(x_trn,y_trn)

		x_tst = features.ix[tst_i]
		y_tst = classes.ix[tst_i]
		y_pred = classifier.predict(x_tst)
		labels = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs'] 
		cm 	=  metrics.confusion_matrix(y_tst, y_pred, labels=labels)
		for i in range (6):
			for j in range(6):
				matrix[i][j] += cm[i][j]
	for d in range(len(matrix)):
		for e in range(len(matrix)):
			matrix[d][e] = matrix[d][e]/10
	return {'matriz' : matrix, 'labels' : labels}
		



kFold = 10
serialFile = 'serial'
sourceFile = 'wisdmTransformedNew.csv'
features = ['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','XAVG','YAVG','ZAVG','XPEAK','YPEAK','ZPEAK','XABSOLDEV','YABSOLDEV','ZABSOLDEV','XSTANDDEV','YSTANDDEV','ZSTANDDEV','RESULTANT']
classes = 'class'
pd_dataset = dataFrame(sourceFile, features, classes)
serial = readFolds(serialFile)

## Criando Folds
#criar_folds(kFold, serialFile, pd_dataset)

## Datasets

meanRemovalFeatures = pd.DataFrame(meanRemoval(pd_dataset['featuresMean']))
normalizationFeatures = pd.DataFrame(normalization(pd_dataset['featuresMean']))
meanRemovalNormalizationFeatures = pd.DataFrame(normalization(meanRemoval(pd_dataset['featuresMean'])))


## Classificadores - Modelos
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier


adaBoostClassifier = AdaBoostClassifier(tree.DecisionTreeClassifier())
decisionTreeClassifier = tree.DecisionTreeClassifier()
logisticRegression = linear_model.LogisticRegression()
mlp = MLPClassifier(early_stopping=True)

print("Matriz de confusao Adaboost")
printData(fitandPredict(pd_dataset['featuresMean'],pd_dataset['classes'] ,serialFile , adaBoostClassifier))

print("Matriz de confusao Decision Tree")
printData(fitandPredict(pd_dataset['featuresMean'],pd_dataset['classes'] ,serialFile , decisionTreeClassifier))
print("Matriz de confusao Logistic Regression")
printData(fitandPredict(pd_dataset['featuresMean'],pd_dataset['classes'] ,serialFile , logisticRegression))
print("Matriz de confusao MLP")
printData(fitandPredict(pd_dataset['featuresMean'],pd_dataset['classes'] ,serialFile , mlp))

print

print("Matriz de confusao Decisio Tree with Mean Removal")
printData(fitandPredict(meanRemovalFeatures,pd_dataset['classes'] ,serialFile , decisionTreeClassifier))
print("Matriz de confusao Logistic Regression with Mean Removal")
printData(fitandPredict(meanRemovalFeatures,pd_dataset['classes'] ,serialFile , logisticRegression))
print("Matriz de confusao MLP with Mean Removal")
printData(fitandPredict(meanRemovalFeatures,pd_dataset['classes'] ,serialFile , mlp))

print

print("Matriz de confusao Adaboost Normalization Features")
printData(fitandPredict(normalizationFeatures,pd_dataset['classes'] ,serialFile , adaBoostClassifier))
print("Matriz de confusao Decision Tree Normalization Features")
printData(fitandPredict(normalizationFeatures,pd_dataset['classes'] ,serialFile , decisionTreeClassifier))
print("Matriz de confusao Logistic Regression Normalization Features")
printData(fitandPredict(normalizationFeatures,pd_dataset['classes'] ,serialFile , logisticRegression))
print("Matriz de confusao MLP Normalization Features")
printData(fitandPredict(normalizationFeatures,pd_dataset['classes'] ,serialFile , mlp))

print


print("Matriz de confusao Adaboost with Mean Removal Normalization")
printData(fitandPredict(meanRemovalNormalizationFeatures,pd_dataset['classes'] ,serialFile , adaBoostClassifier))
print("Matriz de confusao Decision Tree with Mean Removal Normalization")
printData(fitandPredict(meanRemovalNormalizationFeatures,pd_dataset['classes'] ,serialFile , decisionTreeClassifier))
print("Matriz de confusao Logistic Regression with Mean Removal Normalization")
printData(fitandPredict(meanRemovalNormalizationFeatures,pd_dataset['classes'] ,serialFile , logisticRegression))
print("Matriz de confusao MLP with Mean Rremovla Normalization")
printData(fitandPredict(meanRemovalNormalizationFeatures,pd_dataset['classes'] ,serialFile , mlp))