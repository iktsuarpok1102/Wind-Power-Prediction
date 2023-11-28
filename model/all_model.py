from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

def polynomial(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])

def randomfor(n_estimators,max_features):
    return RandomForestRegressor(n_estimators=n_estimators,max_features=max_features)

def SVMRegression(C):
    return SVR(C=C)

def NeuralNet(optimizer):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(17,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model