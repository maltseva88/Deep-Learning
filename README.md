# Deep Learning

In this assignment, I've built and evaluated deep learning models using both the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

In this assignment, you used deep learning recurrent neural networks to model bitcoin closing prices. One model uses the FNG indicators to predict the closing price while the second model used a window of closing prices to predict the nth closing price.

### Preparing the data for training and testing

For the Fear and Greed model, I used the FNG values to try and predict the closing price. 

For the closing price model, I useed previous closing prices to try and predict the next closing price. 

For each model used 70% of the data for training and 30% of the data for testing.

Applied a MinMaxScaler to the X and y values to scale the data for the model.

Finally, reshaped the X_train and X_test values to fit the model's requirement of samples, time steps, and features. (*example:* `X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))`)

### Build and train custom LSTM RNNs

In each Jupyter Notebook, created the same custom LSTM RNN architecture. In one notebook, I fit the data using the FNG values. In the second notebook, I fit the data using only closing prices.

Use the same parameters and training steps for each model. This is necessary to compare each model accurately.

### Evaluate the performance of each model

Finally, use the testing data to evaluate each model and compare the performance.

Use the above to answer the following:


Which model has a lower loss? - RNN CLosing Prices 
Which model tracks the actual values better over time? - RNN Closing Prices
Which window size works best for the model? - window size = 5
