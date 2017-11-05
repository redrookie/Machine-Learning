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

