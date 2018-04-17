from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv1D, MaxPooling1D, Activation
from keras.models import load_model
import keras
import random
import time


""" DATA PREPROCESSING """
## 'time series data' -> 'supervised data'
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

## Data Load & Preprocessing
dataset = read_csv('/Users/kyubum/PycharmProjects/Keras_Sales_Forecast/Final_code/flamingo2.csv', header=0, index_col=0)
values = dataset.values

## String column encoding (string->2,1 ...) <해당사항없음>
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])

## Float as String -> Float
for i in range(len(values)):
    for j in range(len(values[i])):
        if isinstance(values[i][j],str):
            values[i][j] = values[i][j].replace(" ", "")
            values[i][j] = values[i][j].replace(",", "")
            values[i][j] = float(values[i][j])
values = values.astype('float32')

## Normalize features(range : 0~1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

## Specify the number of lag hours
n_hours = 3
n_features = 3  #x의 개수

## Frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)



""" BUILDING & FIT TSA_KERAS MODEL """
bestPram = [29, 82, 'causal', 93, 0.64, 64, 39]

# split into train and test sets
values = reframed.values
n_train_hours = 120
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
#print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(Conv1D(input_shape=(train_X.shape[1], train_X.shape[2]) , filters= bestPram[0], kernel_size= bestPram[1], activation='relu',padding=bestPram[2]))
model.add(MaxPooling1D(pool_size=3))
model.add(LSTM(bestPram[3], dropout=bestPram[4],input_shape=(train_X.shape[1], train_X.shape[2])))  #dropout -> overfitting방지
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=bestPram[5], batch_size=bestPram[6], validation_data=(test_X, test_y), shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


""" FORECAST """
# make a prediction
yhat = model.predict(test_X)  #y의 예측 값(리스트)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -3:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -3:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.figure(figsize=(100, 2))
pyplot.plot(inv_y, label='true value')      #y 실제값
pyplot.plot(inv_yhat, label='prediction value')   #y 예측값
pyplot.legend()
pyplot.ylim((0,2500000))
pyplot.show()