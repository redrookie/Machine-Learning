wisdm = open('wisdmTransformed.csv', 'r')
new = open('wisdmTransformedNew.csv', 'w')

cont = 0;
for linha in wisdm:
	linha = linha.strip()
	if len(linha) != 0:
		cont += 1
		print linha
		linha = linha.replace('?', 'NaN')
		new.write(linha+'\n')
print cont
new.close()
wisdm.close()
