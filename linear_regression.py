import numpy as np
import matplotlib.pyplot as plt
import pandas

# Function that creates the X matrix as defined for fitting our model
def create_X(x,deg):
    X = np.ones((len(x),deg+1))
    for i in range(1,deg+1):
        if(i == 1):
            X[:,i] = 0
        else:
            X[:,i] = x**i
    return X

# Function for predicting the response
def predict_y(x,beta):
    return np.dot(create_X(x,len(beta)-1),beta)

# Function for fitting the model
    
#X = np.array([2, 0, 3])
def fit_beta(df,deg):
    return np.linalg.lstsq(create_X(df.x,deg),df.y,rcond=None)[0]

# Function for computing the MSE
def mse(y,yPred):
    return np.mean((y-yPred)**2)

# Loading training and test data
dfTrain = pandas.read_csv('Data_Train1.csv')
dfTest = pandas.read_csv('Data_Test1.csv')

############ TRAINING A MODEL

# Fitting model
deg = 2
#X = create_X(dfTrain.x,deg)


beta = fit_beta(dfTrain,deg)

# Computing training error
yPredTrain = predict_y(dfTrain.x,beta)
err = mse(dfTrain.y,yPredTrain)
print('Training Error = {:2.3}'.format(err))

# Computing test error
yPredTest = predict_y(dfTest.x,beta)
#err = mse(dfTest.y,yPredTest)
#print('Test Error = {:2.3}'.format(err))

############ PLOTTING FITTED MODEL
x = np.linspace(-5,5,100)
y = predict_y(x,beta)

plt.plot(x,y,'b-',dfTrain.x,dfTrain.y,'r.')
plt.show()