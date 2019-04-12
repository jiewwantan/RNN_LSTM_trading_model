# RNN LSTM Trading Modelling

This project explores stock trading modelling with the use recurrent neural network (RNN) with long-short term memory (LSTM) architecture.  CNN+LSTM hybrid architecture was tried.

## Data preprocessing
Data comprises of the asset's OHLCV data and technical data derive from its OHLCV data. A feature selection algorithm are then used to select the data series based on importance. 
The selected feature data are visualized below. This visualization of pre-processed data demonstrates data richness. 

[image0]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/features_visualization_AAPL.png "Selected features' visualization"
![Selected features' visualization][image0]

## Model Construction & Training
Two machine learning algorithms were tested for price prediction application: A multi-layer recurrent neural network with Long-short term memory (LSTM) cells and a hybrid consists of 2 convolutional neural network layers & 1 LSTM neural network layers.

With CAPM model in mind, future stock price (the next day) is predicted with today and past market (S&P 500) and stock data. Each set of stock and market data consists of indicator time series derived from their respective price and volume data.

Depends on user's computational resource, it may take sometime to train the neural network models. rnn_modelling.py progrma takes about 15 minutes to train on both RNN LSTM and CNN + RNN LSTM models on a machine equipped with Nvidia Geforce 1080Ti. For user convenience, their pre-trained models are included in this repository: best_cnn_model_AAPL.h5 & best_lstm_model_AAPL.h5 for which rnn_modelling.py will skip training and predict from the loaded pre-trained models.

Whereas for rnn_modelling.ipynb notebook, if pre-trained models are available the loading sequence will retrain existing pre-trained models with the hope of improved learning. User may rename the pre-trained models for this notebook to start training from fresh. Or proceed directly to the respecitve model loading cell to get instant prediction.

All training and the pre-trained models in this notebook and rnn_modelling.py are done for stock AAPL. In rnn_modelling.py, user is also given the option to choose other stocks, the program will check if enough historical data is available for training, otherwise user will be prompted to enter a different stock quote.

## Results
### Prediction on unseen data after model training

[image1]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_LSTM_AAPL_train.png "Training LSTM_AAPL"
![Training LSTM_AAPL][image1]

[image2]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_LSTM_IBM.png "Testing LSTM_IBM"
![Testing LSTM_IBM][image2]

[image3]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_CNN_AAPL.png "Testing LSTM_CNN_AAPL"
![Testing LSTM_CNN_AALP][image3]

[image4]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/prediction_CNN_IBM.png "Testing LSTM_CNN_IBM"
![Testing LSTM_CNN_IBM][image4]

### Backtesting result using trained model to generate trade signals

[image5]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/AAPL_trade_signal.png "Backtesting trade signals"
![Backtesting trade signals][image5]

[image6]: https://github.com/jiewwantan/RNN_LSTM_trading_model/blob/master/AAPL_cumreturns.png "Backtesting cummulative returns"
![Backtesting cummulative returns][image6]


## Conclusion
After trying a number of model architectures, optimizers, hyperparameter adjustment and normalization techniques, the unseen data prediction and backtesting results shows that it is possible to use RNN-LSTM for stock trading.  I reckon real world trading result will discount the backtesting result even further. So these models will need to have better generalization ability in order to get pass backtesting and eventually became a workable real world trading algorithm.

For improvements, I suggest:</br>
More context and fundamental data.</br>
Predict returns instead of price.</br>
Try reinforcement learning algorithm
