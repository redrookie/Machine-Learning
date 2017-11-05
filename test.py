from sklearn.model_selection import StratifiedKFold
import pandas as pd
import cPickle as pickle
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict 
import matplotlib.pyplot as plt
import pylab as pl


def dataFrame(dataset, features, classes):
	data = pd.read_csv(dataset)
	x = data[features]
	y = data[classes]
	return {'features': x, 'classes': y}

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
		cm =  metrics.confusion_matrix(y_tst, y_pred, labels=labels)

		print(cm)
		
		'''
		fig = plt.figure()

		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		pl.title('Confusion matrix of the classifier')
		fig.colorbar(cax)
		ax.set_xticklabels([''] + labels)
		ax.set_yticklabels([''] + labels)
		pl.xlabel('Predicted')
		pl.ylabel('True')
		pl.show()
		'''



kFold = 10
serialFile = 'serial'
sourceFile = 'wisdm.csv'
features = ['timestamp','x-acceleration','y-accel','z-accel']
classes = 'activity'
pd_dataset = dataFrame(sourceFile, features, classes)
serial = readFolds(serialFile)

## Criando Folds
#criar_folds(kFold, serialFile, pd_dataset)

## Datasets
meanRemovalFeatures = meanRemoval(pd_dataset['features'])
normalizationFeatures = normalization(pd_dataset['features'])
meanRemovalNormalizationFeatures = normalization(meanRemoval(pd_dataset['features']))



## Classificadores - Modelos
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

decisionTreeClassifier = tree.DecisionTreeClassifier()
logisticRegression = linear_model.LogisticRegression()
mlp = MLPClassifier()

print("Matriz de confusao DT")
fitandPredict(pd_dataset['features'],pd_dataset['classes'] ,serialFile , decisionTreeClassifier)
print("Matriz de confusao LR")
fitandPredict(pd_dataset['features'],pd_dataset['classes'] ,serialFile , logisticRegression)
print("Matriz de confusao MLP")
fitandPredict(pd_dataset['features'],pd_dataset['classes'] ,serialFile , mlp)

print()

print("Matriz de confusao DTMR")
fitandPredict(meanRemovalFeatures,pd_dataset['classes'] ,serialFile , decisionTreeClassifier)
print("Matriz de confusao LRMR")
fitandPredict(meanRemovalFeatures,pd_dataset['classes'] ,serialFile , logisticRegression)
print("Matriz de confusao MLPMR")
fitandPredict(meanRemovalFeatures,pd_dataset['classes'] ,serialFile , mlp)

print()

print("Matriz de confusao DTNF")
fitandPredict(normalizationFeatures,pd_dataset['classes'] ,serialFile , decisionTreeClassifier)
print("Matriz de confusao LRNF")
fitandPredict(normalizationFeatures,pd_dataset['classes'] ,serialFile , logisticRegression)
print("Matriz de confusao MLPNF")
fitandPredict(normalizationFeatures,pd_dataset['classes'] ,serialFile , mlp)

print()

print("Matriz de confusao DTMRN")
fitandPredict(meanRemovalNormalizationFeatures,pd_dataset['classes'] ,serialFile , decisionTreeClassifier)
print("Matriz de confusao LRMRN")
fitandPredict(meanRemovalNormalizationFeatures,pd_dataset['classes'] ,serialFile , logisticRegression)
print("Matriz de confusao MLPMRN")
fitandPredict(meanRemovalNormalizationFeatures,pd_dataset['classes'] ,serialFile , mlp)

## Classificadores - scores

#Decision Tree
decisionTree = cross_val_score(decisionTreeClassifier, pd_dataset['features'], pd_dataset['classes'], cv=serial)
decisionTreeMean = cross_val_score(decisionTreeClassifier, meanRemovalFeatures, pd_dataset['classes'], cv=serial)

#Logistic Regression
logisticRegressionk = cross_val_score(logisticRegression, pd_dataset['features'], pd_dataset['classes'], cv=serial)
logisticRegressionMean = cross_val_score(logisticRegression, meanRemovalFeatures, pd_dataset['classes'], cv=serial)


 
#MLP
mlpk = cross_val_score(mlp, pd_dataset['features'], pd_dataset['classes'], cv=serial)
mlpMean = cross_val_score(mlp, meanRemovalFeatures, pd_dataset['classes'], cv=serial)

'''
##Train and predict for the confusion matrix
#selecting target and shot


shot = pd_dataset['features']
target = pd_dataset['classes']

#spliting data
shot_trn,shot_tst,target_trn,target_tst = train_test_split(shot,target,random_state = 0)

#fitting targets with their shots, using all three methods that were used before
decisionTreeClassifier.fit(shot_trn,target_trn)

logisticRegression.fit(shot_trn,target_trn)
mlp.fit(shot_trn,target_trn)
# predicting methods with theis respectives functions
target_pred_tree= decisionTreeClassifier.predict(shot_tst)
target_pred_log = logisticRegression.predict(shot_tst)
target_pred_mlp = mlp.predict(shot_tst)
print
print
print "Tree accuracy and CMatrix"
print metrics.accuracy_score(target_tst,target_pred_tree) 
print metrics.confusion_matrix(target_tst,target_pred_tree)

print
print
print"LogisticRegression accuracy and CMatrix"
print metrics.accuracy_score(target_tst,target_pred_log)
print metrics.confusion_matrix(target_tst,target_pred_log)

print
print
print "Mlp accuracy and CMatrix"
print metrics.accuracy_score(target_tst,target_pred_mlp)
print metrics.confusion_matrix(target_tst,target_pred_mlp)
'''
