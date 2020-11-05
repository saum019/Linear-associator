#@author Saumya, Bhatt


import numpy as np
from sklearn.metrics import mean_squared_error

class LinearAssociator(object):
    
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"): 
        self.input_dimensions=input_dimensions
        self.number_of_nodes=number_of_nodes
        self.transfer_function=transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None): 
        self.weights = []
        if seed != None:
            np.random.seed(seed)
        self.weights = np.array(self.weights, dtype=np.float)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)
            

    def set_weights(self, W):
        if len(W)==self.number_of_nodes and len(W[0])==self.input_dimensions:
            self.weights=W
            return None 
        else:
            return -1
        
    def get_weights(self):
        self.weights = np.array(self.weights, dtype=np.float)
        return self.weights
    
    def transfer(self,y):
         if self.transfer_function=="Hard_limit":
             self.y=np.where(y[:] <0, 0,1) 
        
         elif self.transfer_function=="Linear" :
             self.y=np.array(y, dtype=np.float)
         return self.y
        
    def predict(self, X):
        multi = np.dot(self.weights,X)
        predicted = self.transfer(multi)
        return predicted
    
    def fit_pseudo_inverse(self, X, y):  
        y=self.transfer(y)
        pseudo_inverse=np.linalg.pinv(X)
        self.weights=np.dot(y,pseudo_inverse)
    
    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        size=len(X[0])   
        if learning.lower()=="delta":
            print("delta")
            j=0
            for i in range(num_epochs):
               for j in range(0,X.shape[1],batch_size):
                  k=j+batch_size
                  if j>size:
                      k=size
                  x_batch=X[:,j:k]
                  y_batch=y[:,j:k]
                  a=self.predict(x_batch)
                  self.weights = np.array(self.weights, dtype=np.float)
                  self.weights = self.weights +(alpha*np.dot(y_batch- a,x_batch.transpose()))   
            print(self.weights)
              
        elif learning.lower()=="filtered":
            print("Filtered")
            for i in range(num_epochs):
               for j in range(0,X.shape[1],batch_size):
                  k=j+batch_size
                  if j>size:
                      k=size
                  x_batch=X[:,j:k]
                  y_batch=y[:,j:k]
                  a=self.predict(x_batch)
                  self.weights = np.array(self.weights, dtype=np.float)
                  self.weights=(1.0-gamma)*self.weights+(alpha*np.dot(y_batch,x_batch.transpose()))
            print(self.weights)
            
        elif learning.lower()=="unsupervised_hebb":
            size=len(X[0])
            print("Unsupervised_hebb")
            j=0
            for i in range(num_epochs):
               for j in range(0,X.shape[1],batch_size):
                  k=j+batch_size
                  if j>size:
                      k=size
                  x_batch=X[:,j:k]
                  a=self.predict(x_batch)
                  self.weights = np.array(self.weights, dtype=np.float)
                  self.weights=self.weights+(alpha*np.dot(a,x_batch.transpose()))
            print(self.weights)
       
        else:
            print("Enter Valid Learning Rule!!")
         
    def calculate_mean_squared_error(self, X, y):
        a=self.predict(X)
        Mean_squared_error=mean_squared_error(y, a)
        return Mean_squared_error
    

            
