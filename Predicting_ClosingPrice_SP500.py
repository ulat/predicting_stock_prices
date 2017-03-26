#!/usr/bin/env python
# coding: utf-8

# # Predicting Stock Prices
#
# Code generated from Youtube channel of Siraj Raval. See link: https://www.youtube.com/watch?v=ftMq5ps503w
#
# Prediction closing price of S&P500 using LSTM network

# In[2]:

if __name__ == "__main__":

    from keras.layers.core import Dense, Activation, Dropout
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential
    import lstm
    import time
    # helper libraries

    print ("Predicting Stock Prices started")
    X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 200, True)
    batch_size = 1

    model = Sequential()

    model.add(LSTM(
        input_dim=1,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        input_dim=50,
        output_dim=100,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        input_dim=50,
        output_dim=100,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        input_dim=50,
        output_dim=100,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print ('compilation time : ', time.time() - start)

    # ## Train the model

    model.fit(
        X_train,
        y_train,
        batch_size=100,
        nb_epoch=1,
        validation_split=0.1)

    # save model after training
    model_filename = '/output/preditc_sp500.h5'
    model.save(model_filename)

    # ## Make some predictions
