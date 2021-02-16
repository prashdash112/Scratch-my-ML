import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class linear_regression:
    
    '''
    A linear model makes a prediction by simply computing a weighted
    sum of the input features, plus a constant called the bias term.
    
    To find the value of θ that minimizes the cost function, there is a closed-form solution, 
    a mathematical equation that gives the result directly. This is called
    the Normal Equation.
    
    Initializing the linear regression model with variables.........
    Use the object of linear regression class for prediction
    for ex: model=linear_regression()

    '''

    
    def __init__(self):
        
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def fit(self,x_train,y_train):
        
        '''
        
        Fiting the model means fitting a line to the data points which approximately explains the deviation of points & can act as a good predictor for future input values.
        The best optimized line or data points can be obtained through the normal equation.
        
        x_train is the training input features aka independent variable
        y_train is the training output feature aka dependent variable
        
        Using the relation between input & output, a predictor line(best optimized theta values) is obtained for prediction.
        
        '''
        
        global theta
        
        if ((isinstance(x_train,np.ndarray)==True) & (isinstance(y_train,np.ndarray)==True)):
            theta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        else:
            #X_train=x_train.values
            #Y_train=y_train.values
            theta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
            
        print('.........Model trained.......')
        return theta
    
    def predict(self,x_test):
        
        '''
        
        Predict method takes the testing dataframe & produces the predicted values which corresponds to the input.
        x_test is testing independent features.
        
        '''
        #x_test = x_test.values
        #x_test=x_test.values
        #X_test = np.c_[np.ones((x_test.shape[0], x_test.shape[1])), x_test] # add x0 = 1 to each instance
        
        if (isinstance(x_test,np.ndarray)==True):
            y_predict = x_test.dot(theta)
            return y_predict
        else:
            y_predict = x_test.dot(theta)
            return y_predict.values

if __name__ == '__main__':

	df = pd.read_csv(r'C:\Users\Prashant\Desktop\notebooks\11-Linear-Regression\USA_Housing.csv')
	X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
	y = df[['Price']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	model = linear_regression()
	print(model.fit(X_train,y_train))
	print('\n\n')
	predictions=model.predict(X_test)
	print(predictions)
	print('\n\n')
	pred = pd.DataFrame(data=predictions,columns=['prediction'])
	y_test = pd.DataFrame(data=y_test['Price'].values,columns=['Price'])
	final_result = pd.concat([y_test,pred],axis=1)
	final_result['error'] = final_result['Price']-final_result['prediction']
	print(final_result)
