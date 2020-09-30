# LSTM Stock Predictor Using Closing Prices

In this notebook, you will build and train a custom LSTM RNN that uses a 10 day window of Bitcoin closing prices to predict the 11th day closing price. 

You will need to:

1. Prepare the data for training and testing
2. Build and train a custom LSTM RNN
3. Evaluate the performance of the model

## Data Preparation

In this section, you will need to prepare the training and testing data for the model. The model will use a rolling 10 day window to predict the 11th day closing price.

You will need to:
1. Use the `window_data` function to generate the X and y values for the model.
2. Split the data into 70% training and 30% testing
3. Apply the MinMaxScaler to the X and y values
4. Reshape the X_train and X_test data for the model. Note: The required input format for the LSTM is:

```python
reshape((X_train.shape[0], X_train.shape[1], 1))
```


```python
import numpy as np
import pandas as pd
import hvplot.pandas
```










```python
# Set the random seed for reproducibility
# Note: This is for the homework solution, but it is good practice to comment this out and run multiple experiments to evaluate your model
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
```


```python
# Load the fear and greed sentiment data for Bitcoin
df = pd.read_csv('btc_sentiment.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
df = df.drop(columns="fng_classification")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fng_value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-29</th>
      <td>19</td>
    </tr>
    <tr>
      <th>2019-07-28</th>
      <td>16</td>
    </tr>
    <tr>
      <th>2019-07-27</th>
      <td>47</td>
    </tr>
    <tr>
      <th>2019-07-26</th>
      <td>24</td>
    </tr>
    <tr>
      <th>2019-07-25</th>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Load the historical closing prices for Bitcoin
df2 = pd.read_csv('btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
df2 = df2.sort_index()
df2.tail()
```




    Date
    2019-07-25    9882.429688
    2019-07-26    9847.450195
    2019-07-27    9478.320313
    2019-07-28    9531.769531
    2019-07-29    9529.889648
    Name: Close, dtype: float64




```python
# Join the data into a single DataFrame
df = df.join(df2, how="inner")
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fng_value</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-25</th>
      <td>42</td>
      <td>9882.429688</td>
    </tr>
    <tr>
      <th>2019-07-26</th>
      <td>24</td>
      <td>9847.450195</td>
    </tr>
    <tr>
      <th>2019-07-27</th>
      <td>47</td>
      <td>9478.320313</td>
    </tr>
    <tr>
      <th>2019-07-28</th>
      <td>16</td>
      <td>9531.769531</td>
    </tr>
    <tr>
      <th>2019-07-29</th>
      <td>19</td>
      <td>9529.889648</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fng_value</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-02-01</th>
      <td>30</td>
      <td>9114.719727</td>
    </tr>
    <tr>
      <th>2018-02-02</th>
      <td>15</td>
      <td>8870.820313</td>
    </tr>
    <tr>
      <th>2018-02-03</th>
      <td>40</td>
      <td>9251.269531</td>
    </tr>
    <tr>
      <th>2018-02-04</th>
      <td>24</td>
      <td>8218.049805</td>
    </tr>
    <tr>
      <th>2018-02-05</th>
      <td>11</td>
      <td>6937.080078</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y

def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)
```


```python
# Predict Closing Prices using a 10 day window of previous closing prices
# Then, experiment with window sizes anywhere from 1 to 10 and see how the model performance changes
window_size = 5

# Column index 0 is the 'fng_value' column
# Column index 1 is the `Close` column

feature_column = 1
target_column = 1
X, y = window_data(df, window_size, feature_column, target_column)
```


```python
# Use 70% of the data for training and the remaineder for testing

split = int(0.7 * len(X))

X_train = X[: split]
X_test = X[split:]

y_train = y[: split]
y_test = y[split:]
```


```python
from sklearn.preprocessing import MinMaxScaler

# Use the MinMaxScaler to scale data between 0 and 1.

scaler = MinMaxScaler()

# Fit the MinMaxScaler object with the features data X
scaler.fit(X)

# Scale the features training and testing sets
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Fit the MinMaxScaler object with the target data Y

scaler.fit(y)

# Scale the target training and testing sets

y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
```


```python
# Reshape the features for the model

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Print some sample data after reshaping the datasets

print (f"X_train sample values:\n{X_train[:3]} \n")
print (f"X_test sample values:\n{X_test[:3]}")

```

    X_train sample values:
    [[[0.60761794]
      [0.58242373]
      [0.62172321]
      [0.51499412]
      [0.38267307]]
    
     [[0.58242373]
      [0.62172321]
      [0.51499412]
      [0.38267307]
      [0.46160996]]
    
     [[0.62172321]
      [0.51499412]
      [0.38267307]
      [0.46160996]
      [0.4503991 ]]] 
    
    X_test sample values:
    [[[0.03856719]
      [0.03679562]
      [0.03728835]
      [0.03974167]
      [0.04528668]]
    
     [[0.03679562]
      [0.03728835]
      [0.03974167]
      [0.04528668]
      [0.04528668]]
    
     [[0.03728835]
      [0.03974167]
      [0.04528668]
      [0.04528668]
      [0.07024855]]]


---

## Build and Train the LSTM RNN

In this section, you will design a custom LSTM RNN and fit (train) it using the training data.

You will need to:
1. Define the model architecture
2. Compile the model
3. Fit the model to the training data

### Hints:
You will want to use the same model architecture and random seed for both notebooks. This is necessary to accurately compare the performance of the FNG model vs the closing price model. 


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```


```python
# Build the LSTM model. 
# The return sequences need to be set to True if you are adding additional LSTM layers, but 
# You don't have to do this for the final layer. 
# Note: The dropouts help prevent overfitting
# Note: The input shape is the number of time steps and the number of indicators
# Note: Batching inputs has a different input shape of Samples/TimeSteps/Features

model = Sequential()

number_units = 30
dropout_fraction = 0.2

# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))

# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))


# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))


# Output layer
model.add(Dense(1))
```


```python
# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")
```


```python
# Summarize the model
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_3 (LSTM)                (None, 5, 30)             3840      
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 5, 30)             0         
    _________________________________________________________________
    lstm_4 (LSTM)                (None, 5, 30)             7320      
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 5, 30)             0         
    _________________________________________________________________
    lstm_5 (LSTM)                (None, 30)                7320      
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 30)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 31        
    =================================================================
    Total params: 18,511
    Trainable params: 18,511
    Non-trainable params: 0
    _________________________________________________________________



```python
# Train the model
# Use at least 10 epochs
# Do not shuffle the data
# Experiement with the batch size, but a smaller batch size is recommended
model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=1, verbose=1)
```

    Epoch 1/10
    375/375 [==============================] - 6s 17ms/step - loss: 0.0240
    Epoch 2/10
    375/375 [==============================] - 7s 19ms/step - loss: 0.0145
    Epoch 3/10
    375/375 [==============================] - 7s 18ms/step - loss: 0.0172
    Epoch 4/10
    375/375 [==============================] - 5s 13ms/step - loss: 0.0190
    Epoch 5/10
    375/375 [==============================] - 4s 12ms/step - loss: 0.0169
    Epoch 6/10
    375/375 [==============================] - 5s 14ms/step - loss: 0.0145
    Epoch 7/10
    375/375 [==============================] - 5s 14ms/step - loss: 0.0111
    Epoch 8/10
    375/375 [==============================] - 7s 18ms/step - loss: 0.0105
    Epoch 9/10
    375/375 [==============================] - 6s 17ms/step - loss: 0.0107
    Epoch 10/10
    375/375 [==============================] - 6s 17ms/step - loss: 0.0090





    <tensorflow.python.keras.callbacks.History at 0x7fdbb0ff5950>



---

## Model Performance

In this section, you will evaluate the model using the test data. 

You will need to:
1. Evaluate the model using the `X_test` and `y_test` data.
2. Use the X_test data to make predictions
3. Create a DataFrame of Real (y_test) vs predicted values. 
4. Plot the Real vs predicted values as a line chart

### Hints
Remember to apply the `inverse_transform` function to the predicted and y_test values to recover the actual closing prices.


```python
# Evaluate the model
model.evaluate(X_test, y_test)
```

    6/6 [==============================] - 0s 7ms/step - loss: 0.0459





    0.04593052342534065




```python
# Make some predictions
predicted = model.predict(X_test)
```


```python
# Recover the original prices instead of the scaled version

predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
```


```python
# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
stocks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Real</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-18</th>
      <td>3670.919922</td>
      <td>3807.168457</td>
    </tr>
    <tr>
      <th>2019-02-19</th>
      <td>3912.570068</td>
      <td>3815.262451</td>
    </tr>
    <tr>
      <th>2019-02-20</th>
      <td>3924.239990</td>
      <td>3843.646484</td>
    </tr>
    <tr>
      <th>2019-02-21</th>
      <td>3974.050049</td>
      <td>3887.665771</td>
    </tr>
    <tr>
      <th>2019-02-22</th>
      <td>3937.040039</td>
      <td>3940.775879</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the real vs predicted values as a line chart
stocks.plot(title="Real vs Predicted Values")
```




    <AxesSubplot:title={'center':'Real vs Predicted Values'}>




![png](output_26_1.png)



```python

```


```python

```
