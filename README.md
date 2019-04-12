# RNN LSTM Trading Modelling

This project explores stock trading modelling with the use recurrent neural network (RNN) with long-short term memory (LSTM) architecture.  CNN+LSTM hybrid architecture was tried.

## Data preprocessing
[image1]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/features_visualization_AAPL.png "Selected features' visualization"
![Selected features' visualization][image1]

## Model Construction & Training
Two machine learning algorithms were tested for price prediction application: A multi-layer recurrent neural network with Long-short term memory (LSTM) cells and a hybrid consists of 2 convolutional neural network layers & 1 LSTM neural network layers.

With CAPM model in mind, future stock price (the next day) is predicted with today and past market (S&P 500) and stock data. Each set of stock and market data consists of indicator time series derived from their respective price and volume data.

Depends on user's computational resource, it may take sometime to train the neural network models. MP2.py progrma takes about 15 minutes to train on both RNN LSTM and CNN + RNN LSTM models on a machine equipped with Nvidia Geforce 1080Ti. For user convenience, their pre-trained models are included in this repository: best_cnn_model_AAPL.h5 & best_lstm_model_AAPL.h5 for which MP2.py will skip training and predict from the loaded pre-trained models.

Whereas for MP2.ipynb notebook, if pre-trained models are available the loading sequence will retrain existing pre-trained models with the hope of improved learning. User may rename the pre-trained models for this notebook to start training from fresh. Or proceed directly to the respecitve model loading cell to get instant prediction.

All training and the pre-trained models in this notebook and MP2.py are done for stock AAPL. In MP2.py, user is also given the option to choose other stocks, the program will check if enough historical data is available for training, otherwise user will be prompted to enter a different stock quote.

## Results
### Prediction Plots

[image1]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_LSTM_AAPL_train.png "Training LSTM_AAPL"
![Training LSTM_AAPL][image1]

[image2]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_LSTM_IBM.png "Testing LSTM_IBM"
![Testing LSTM_IBM][image2]

[image3]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_CNN_AAPL.png "Testing LSTM_CNN_AAPL"
![Testing LSTM_CNN_AALP][image3]

[image4]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_CNN_IBM.png "Testing LSTM_CNN_IBM"
![Testing LSTM_CNN_IBM][image4]

## Conclusion
After trying a number of model architectures, optimizers, hyperparameter adjustment and normalization techniques, the results shows that it is possible to predict stock price. Perhaps, backtesting will be required to see if these models can make profitiable trading system. I reckon real world trading result will discount the backtesting result even further. So these models will need to have better generalization ability in order to get pass backtesting and eventually became a workable real wolrd trading algorithm.

For improvements, I suggest:
More context and fundamental data.
Predict returns instead of price.
Try reinforcement learning algorithm
