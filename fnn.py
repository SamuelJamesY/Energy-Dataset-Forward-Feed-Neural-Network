import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import random 

def load_data():
	'''
	Load the data, rename columns, scale input variables
	'''
	df = pd.read_excel('energy.xlsx')
	df.columns = ['compactness','SA','WA','RA','height','orientation','GA','GAD','heating','cooling']
	scaler = MinMaxScaler()
	df.iloc[:,0:8] = scaler.fit_transform(df.iloc[:,0:8])
	return df

def train_test_data_split(data_frame,expnum):
	'''
	create a 60/40 train test split
	'''
	X = data_frame.iloc[:,0:8].to_numpy()
	Y = data_frame.iloc[:,-2:].to_numpy()
	xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.4,random_state=expnum)
	return xtrain,xtest,ytrain,ytest

def neural_network(xtrain,xtest,ytrain,ytest,lr,hidden,expnum):
	'''
	Build and fit a neural network regression model and calculate RMSE and R2 score 
	'''
	nn = MLPRegressor(hidden_layer_sizes = (hidden,hidden),solver='adam',learning_rate_init=lr,random_state=expnum,max_iter=1000)
	nn.fit(xtrain,ytrain)
	ypred = nn.predict(xtest)
	rmse = np.sqrt(mean_squared_error(ypred[:,0],ytest[:,0]))
	r2 = r2_score(ypred[:,0],ytest[:,0])
	return rmse,r2

def linear_regression_model(xtrain,xtest,ytrain,ytest):
	'''
	Use a regular linear regression model
	'''
	regr = linear_model.LinearRegression()
	regr.fit(xtrain,ytrain[:,0])
	ypred = regr.predict(xtest)
	rmse = np.sqrt(mean_squared_error(ypred,ytest[:,0]))
	r2 = r2_score(ypred,ytest[:,0])
	return rmse,r2

def main():
	lr = 0.001
	hidden = 30	
	df = load_data()
	maxexps = 10	
	rmsenn_lst = np.empty(maxexps)
	r2nn_lst = np.empty(maxexps)
	rmselr_lst = np.empty(maxexps)
	r2lr_lst = np.empty(maxexps)
	for exp in range(maxexps):
		xtrain,xtest,ytrain,ytest = train_test_data_split(df,exp)
		rmsenn_lst[exp], r2nn_lst[exp] = neural_network(xtrain,xtest,ytrain,ytest,lr,hidden,exp)
		rmselr_lst[exp], r2lr_lst[exp] = linear_regression_model(xtrain,xtest,ytrain,ytest)
	print('RMSE Neural Net mean:',rmsenn_lst.mean(),'RMSE Neural Net std:', rmsenn_lst.std())
	print('R2 Neural Net mean:',r2nn_lst.mean(),'R2 Neural Net std:', r2nn_lst.std())
	print('RMSE Linear Regression mean:',rmselr_lst.mean(),'RMSE Linear Regression std:', rmselr_lst.std())
	print('R2 Linear Regression std:',r2lr_lst.mean(),'R2 Linear Regression std:', r2lr_lst.std())
	# observation: the neural network did a lot better than the standard linear reg model
if __name__ == '__main__':
	main()