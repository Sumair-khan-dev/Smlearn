"""
Author : Sumair Khan
Version : 0.0.1
Uses : Currently work for LinearRegression

"""
import pandas as pd
import numpy as np

class LinearRegression:
	def __init__(self, *args , **kwargs):
		"""Class of linear regression.It will take data as either a DataFrame or X/Y array and separatethe X and target Y"""
		self.args = args
		self.kwargs = kwargs
		self.intercept = None 
		self.coefficient = None 
		if len(self.args) == 1:
			if isinstance(self.args[0], pd.DataFrame):
				X = self.args[0].iloc[: , :-1].values.tolist()
				Y = self.args[0].iloc[: , -1].values.tolist()
				self.X = np.array(X)
				self.Y = np.array(Y).reshape(-1 , 1)
		
		elif isinstance(self.kwargs.get("X"),list) and isinstance(self.kwargs.get("Y"),list):
			self.X = np.array(self.kwargs.get("X"))
			self.Y = np.array(self.kwargs.get("Y")).reshape(-1, 1)
		elif len(self.args) >= 2:
			self.X = np.array(self.args[0])
			self.Y = np.array(self.args[1]).reshape(-1,1)
		else:
			raise ValueError("Provide either DataFrame or x/y array")
	
	def fit(self):
		"""Compute Theta using Normal Equation"""
		x_b = np.c_[np.ones((self.X.shape[0],1)),self.X]
		
		try:
			theta = np.linalg.inv(x_b.T @ x_b) @ (x_b.T @ self.Y)
		except np.linalg.LinAlgError: 
			theta = np.linalg.pinv(x_b) @ self.Y
			
		self.intercept = theta[0][0]
		self.coefficient = theta[1:]
		
	def predict(self,new_data):
				"""Make prediction from new data"""
				X_new = np.array(new_data)
				
				return self.intercept + X_new @ self.coefficient.flatten()
				

class PolynomailRegression(LinearRegression):
	
	def __init__(self,degree,*args,**kwargs):
		"""Increase the data to given degree for polynomial Regression"""
		self.degree = degree 
		super().__init__(*args,**kwargs)
		X = self.X
		if X.ndim == 1:
			X = X.reshape(-1,1)
		
		features = [X]
		for d in range(2,degree+1):
			features.append(X ** d)
			
		self.X = np.hstack(features)
		
	def predict(self,new_data):
		"""Predict using new data(X) and make that also a poly Features"""
		x_new = np.array(new_data)
		if x_new.ndim == 1:
			x_new = x_new.reshape(-1,1)
		
		features = [x_new]
		for d in range(2,self.degree+1):
			features.append(x_new ** d)
		poly_newx = np.hstack(features)
		
		return self.intercept + poly_newx @ self.coefficient.flatten()
