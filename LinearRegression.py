# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:00:54 2023

@author: engke
"""
class linearRegression():
    import numpy as np
    import pandas as pd
    
    def __init__(self, linear_type='OLS'):
        self.type_ = linear_type
        self.coef_ = []
        self.intercept_ = 0
        self.n_columns = 0
        
    def fit(self, X_train, y_train):
        
        try:
            self.n_columns = len(X_train.columns)
        except:
            self.n_columns = 1
            
        if self.type_ == 'OLS':
            if self.n_columns == 1:
                # simple linear regression
                sx,sy,sxy,sxyz = [],[],[],[]

                for i in range(len(X_train)):
                    sx.append(X_train.values[i] - np.mean(X_train))
                    sy.append(y_train.values[i] - np.mean(y_train))
                    sxy.append(sx[i] * sy[i])
                    sxyz.append(sx[i]**2)

                self.coef_.append(np.sum(sxy) / np.sum(sxyz))
                self.intercept_ = (np.mean(y_train) - self.coef_[0] * np.mean(X_train))
                
            else:
                # multiple linear regression
                sy,sxy,sxyz = [],[],[]
                
                X_train = np.array(X_train).T
                
                for i in range(0, self.n_columns):
                    v = 'x_' + str(i)
                    vars()[v] = [x - X_train[i].mean() for x in X_train[i]]           
                    Xi_trains = vars()[v]
                    
                    for j in range(len(X_train)):
                    
                        sy.append(y_train.values[i] - np.mean(y_train))
                        sxy.append(Xi_trains[i] * sy[i])
                        sxyz.append(Xi_trains[i]**2) 
                    
                    self.coef_.append(np.sum(sxy) / np.sum(sxyz))                   
                    self.intercept_ = [np.mean(y_train) + x*np.mean(X_train) for x in self.coef_]
                
                mses =[]
                x1,x2 = self.intercept_[0], self.intercept_[1]
                
                for i in range(0,len(self.intercept_)):
                    self.intercept_ = x1 if i == 0 else x2
                    y_hat = self.predict(X_train)
                    mses.append(self.mse(y_hat))
                print(mses)
                    
                #self.intercept_ = np.mean(y_train) - np.mean(self.coef_) * np.mean(X_train)
                
        elif self.type_ == 'SGD':
            
            pass
        
        else:
            print('Model not deploy')

        
    def predict(self, X_test):
        y_pred = []
        
        if self.n_columns == 1:
            if type(X_test) is pd.DataFrame:
                X_test = X_test.values

                #y_hat = a*x + b
                for i in range(len(X_test)):
                    y_pred.append(self.coef_[0] * X_test[i] + self.intercept_)   

                return [x[0] for x in y_pred]
            
            else:
                #y_hat = a*x + b
                y_pred = [self.coef_[0] * x + self.intercept_ for x in X_test]   
                return y_pred              

        else:
            
            #y_hat = a*x + b
            for i in range(0, len(X_test)):
                b = modelo.intercept_
                print(b)
                for j in range(0, self.n_columns):
                    b += X.iloc[i][j] * modelo.coef_[j]
                y_pred.append(b)

            return y_pred
        
    
    def scores(self, y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    
    def mse(self, y):  
        return np.sum([(yi - np.mean(y))**2 for yi in y])