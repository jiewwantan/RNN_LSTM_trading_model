import numpy as np
import pandas as pd
import math
import h5py
import fix_yahoo_finance as yf
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as pdr
from time import sleep
from datetime import datetime as dt
import talib as tb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler


# Libraries required by FeatureSelector()
import lightgbm as lgb
import gc
from itertools import chain
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import copy
from matplotlib.dates import (DateFormatter, WeekdayLocator, DayLocator, MONDAY)

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU, GlobalAveragePooling1D
from keras import regularizers
from keras import backend as K
# ------------------------- GLOBAL PARAMETERS -------------------------

# Range of date
START = dt(2000, 1, 1)
END = dt(2018, 12, 7)
PREDICTION_AHEAD = 1
TRAIN_PORTION = 0.9
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
tbd = TensorBoard(log_dir='./tensorboard_logs', histogram_freq=1, embeddings_freq=1)
# ------------------------------ CLASSES ---------------------------------

class FeatureSelector():
    """
    Courtesy of William Koehrsen from Feature Labs
    Class for performing feature selection for machine learning or data preprocessing.

    Implements five different methods to identify features for removal

        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find collinear variables with a correlation greater than a specified correlation coefficient
        4. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
        5. Find low importance features that do not contribute to a specified cumulative feature importance from the gbm

    Parameters
    --------
        data : dataframe
            A dataset with observations in the rows and features in the columns

        labels : array or series, default = None
            Array of labels for training the machine learning model to find feature importances. These can be either binary labels
            (if task is 'classification') or continuous targets (if task is 'regression').
            If no labels are provided, then the feature importance based methods are not available.

    Attributes
    --------

    ops : dict
        Dictionary of operations run and features identified for removal

    missing_stats : dataframe
        The fraction of missing values for all features

    record_missing : dataframe
        The fraction of missing values for features with missing fraction above threshold

    unique_stats : dataframe
        Number of unique values for all features

    record_single_unique : dataframe
        Records the features that have a single unique value

    corr_matrix : dataframe
        All correlations between all features in the data

    record_collinear : dataframe
        Records the pairs of collinear variables with a correlation coefficient above the threshold

    feature_importances : dataframe
        All feature importances from the gradient boosting machine

    record_zero_importance : dataframe
        Records the zero importance features in the data according to the gbm

    record_low_importance : dataframe
        Records the lowest importance features not needed to reach the threshold of cumulative importance according to the gbm


    Notes
    --------

        - All 5 operations can be run with the `identify_all` method.
        - If using feature importances, one-hot encoding is used for categorical variables which creates new columns

    """

    def __init__(self, data, labels=None):

        # Dataset and optional training labels
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided. Feature importance based methods are not available.')

        self.base_features = list(data.columns)
        self.one_hot_features = None

        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None

        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None

        # Dictionary to hold removal operations
        self.ops = {}

        self.one_hot_correlated = False

    def identify_missing(self, missing_threshold):
        """Find the features with a fraction of missing values above `missing_threshold`"""

        self.missing_threshold = missing_threshold

        # Calculate the fraction of missing in each column
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending=False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns=
            {'index': 'feature', 0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.ops['missing'] = to_drop

        print('%d features with greater than %0.2f missing values.\n' % (
        len(self.ops['missing']), self.missing_threshold))

    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={'index': 'feature',
                     0: 'nunique'})

        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))

    def identify_collinear(self, correlation_threshold, one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.

        Using code adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

        Parameters
        --------

        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying correlation features

        one_hot : boolean, default = False
            Whether to one-hot encode the features before calculating the correlation coefficients

        """

        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot

        # Calculate the correlations between every column
        if one_hot:

            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

            corr_matrix = pd.get_dummies(features).corr()

        else:
            corr_matrix = self.data.corr()

        self.corr_matrix = corr_matrix

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index=True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
        len(self.ops['collinear']), self.correlation_threshold))

    def identify_zero_importance(self, task, eval_metric=None,
                                 n_iterations=10, early_stopping=True):
        """

        Identify the features with zero importance according to a gradient boosting machine.
        The gbm can be trained with early stopping using a validation set to prevent overfitting.
        The feature importances are averaged over `n_iterations` to reduce variance.

        Uses the LightGBM implementation (http://lightgbm.readthedocs.io/en/latest/index.html)

        Parameters
        --------

        eval_metric : string
            Evaluation metric to use for the gradient boosting machine for early stopping. Must be
            provided if `early_stopping` is True

        task : string
            The machine learning task, either 'classification' or 'regression'

        n_iterations : int, default = 10
            Number of iterations to train the gradient boosting machine

        early_stopping : boolean, default = True
            Whether or not to use early stopping with a validation set when training


        Notes
        --------

        - Features are one-hot encoded to handle the categorical variables before training.
        - The gbm is not optimized for any particular task and might need some hyperparameter tuning
        - Feature importances, including zero importance features, can change across runs

        """

        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                             "l2" for regression.""")

        if self.labels is None:
            raise ValueError("No training labels provided.")

        # One hot encoding
        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]

        # Add one hot encoded data to original data
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1,))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        print('Training Gradient Boosting Model\n')

        # Iterate through each fold
        for _ in range(n_iterations):

            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=0)

            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=0)

            else:
                raise ValueError('Task must be either "classification" or "regression"')

            # If training using early stopping need a validation set
            if early_stopping:

                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels,
                                                                                              test_size=0.15)
                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric=eval_metric,
                          eval_set=[(valid_features, valid_labels)],
                          early_stopping_rounds=100, verbose=0)

                # Clean up memory
                gc.enable()
                del train_features, train_labels, valid_features, valid_labels
                gc.collect()

            else:
                model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances[
            'importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop

        print('\n%d features with zero importance after one-hot encoding.\n' % len(self.ops['zero_importance']))

    def identify_low_importance(self, cumulative_importance):
        """
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance from the gradient boosting machine. As an example, if cumulative
        importance is set to 0.95, this will retain only the most important features needed to
        reach 95% of the total feature importance. The identified features are those not needed.

        Parameters
        --------
        cumulative_importance : float between 0 and 1
            The fraction of cumulative importance to account for

        """

        self.cumulative_importance = cumulative_importance

        # The feature importances need to be calculated before running
        if self.feature_importances is None:
            raise NotImplementedError("""Feature importances have not yet been determined. 
                                         Call the `identify_zero_importance` method first.""")

        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[
            self.feature_importances['cumulative_importance'] > cumulative_importance]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop

        print('%d features required for cumulative importance of %0.2f after one hot encoding.' % (
        len(self.feature_importances) -
        len(self.record_low_importance), self.cumulative_importance))
        print('%d features do not contribute to cumulative importance of %0.2f.\n' % (len(self.ops['low_importance']),
                                                                                      self.cumulative_importance))

    def identify_all(self, selection_params):
        """
        Use all five of the methods to identify features to remove.

        Parameters
        --------

        selection_params : dict
           Parameters to use in the five feature selection methhods.
           Params must contain the keys ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']

        """

        # Check for all required parameters
        for param in ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']:
            if param not in selection_params.keys():
                raise ValueError('%s is a required parameter for this method.' % param)

        # Implement each of the five methods
        self.identify_missing(selection_params['missing_threshold'])
        self.identify_single_unique()
        self.identify_collinear(selection_params['correlation_threshold'])
        self.identify_zero_importance(task=selection_params['task'], eval_metric=selection_params['eval_metric'])
        self.identify_low_importance(selection_params['cumulative_importance'])

        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)

        print('%d total features out of %d identified for removal after one-hot encoding.\n' % (self.n_identified,
                                                                                                self.data_all.shape[1]))

    def check_removal(self, keep_one_hot=True):

        """Check the identified features before removal. Returns a list of the unique features identified."""

        self.all_identified = set(list(chain(*list(self.ops.values()))))
        print('Total of %d features identified for removal' % len(self.all_identified))

        if not keep_one_hot:
            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:
                one_hot_to_remove = [x for x in self.one_hot_features if x not in self.all_identified]
                print('%d additional one-hot features can be removed' % len(one_hot_to_remove))

        return list(self.all_identified)

    def remove(self, methods, keep_one_hot=True):
        """
        Remove the features from the data according to the specified methods.

        Parameters
        --------
            methods : 'all' or list of methods
                If methods == 'all', any methods that have identified features will be used
                Otherwise, only the specified methods will be used.
                Can be one of ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
            keep_one_hot : boolean, default = True
                Whether or not to keep one-hot encoded features

        Return
        --------
            data : dataframe
                Dataframe with identified features removed


        Notes
        --------
            - If feature importances are used, the one-hot encoded columns will be added to the data (and then may be removed)
            - Check the features that will be removed before transforming data!

        """

        features_to_drop = []

        if methods == 'all':

            # Need to use one-hot encoded data as well
            data = self.data_all

            print('{} methods have been run\n'.format(list(self.ops.keys())))

            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))

        else:
            # Need to use one-hot encoded data as well
            if 'zero_importance' in methods or 'low_importance' in methods or self.one_hot_correlated:
                data = self.data_all

            else:
                data = self.data

            # Iterate through the specified methods
            for method in methods:

                # Check to make sure the method has been run
                if method not in self.ops.keys():
                    raise NotImplementedError('%s method has not been run' % method)

                # Append the features identified for removal
                else:
                    features_to_drop.append(self.ops[method])

            # Find the unique features to drop
            features_to_drop = set(list(chain(*features_to_drop)))

        features_to_drop = list(features_to_drop)

        if not keep_one_hot:

            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:

                features_to_drop = list(set(features_to_drop) | set(self.one_hot_features))

        # Remove the features and return the data

        data = data.drop(features_to_drop, axis=1)
        self.removed_features = features_to_drop

        if not keep_one_hot:
            print('Removed %d features including one-hot features.' % len(features_to_drop))
        else:
            print('Removed %d features.' % len(features_to_drop))

        return data

    def plot_missing(self):
        """Histogram of missing fraction in each feature"""
        if self.record_missing is None:
            raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")

        self.reset_plot()

        # Histogram of missing values
        plt.style.use('seaborn-white')
        plt.figure(figsize=(7, 5))
        plt.hist(self.missing_stats['missing_fraction'], bins=np.linspace(0, 1, 11), edgecolor='k', color='red',
                 linewidth=1.5)
        plt.xticks(np.linspace(0, 1, 11));
        plt.xlabel('Missing Fraction', size=14);
        plt.ylabel('Count of Features', size=14);
        plt.title("Fraction of Missing Values Histogram", size=16);

    def plot_unique(self):
        """Histogram of number of unique values in each feature"""
        if self.record_single_unique is None:
            raise NotImplementedError('Unique values have not been calculated. Run `identify_single_unique`')

        self.reset_plot()

        # Histogram of number of unique values
        self.unique_stats.plot.hist(edgecolor='k', figsize=(7, 5))
        plt.ylabel('Frequency', size=14);
        plt.xlabel('Unique Values', size=14);
        plt.title('Number of Unique Values Histogram', size=16);

    def plot_collinear(self, plot_all=False):
        """
        Heatmap of the correlation values. If plot_all = True plots all the correlations otherwise
        plots only those features that have a correlation above the threshold

        Notes
        --------
            - Not all of the plotted correlations are above the threshold because this plots
            all the variables that have been idenfitied as having even one correlation above the threshold
            - The features on the x-axis are those that will be removed. The features on the y-axis
            are the correlated features with those on the x-axis

        Code adapted from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
        """

        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')

        if plot_all:
            corr_matrix_plot = self.corr_matrix
            title = 'All Correlations'

        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])),
                                                    list(set(self.record_collinear['drop_feature']))]

            title = "Correlations Above Threshold"

        f, ax = plt.subplots(figsize=(10, 8))

        # Diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                    linewidths=.25, cbar_kws={"shrink": 0.6})

        # Set the ylabels
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size=int(160 / corr_matrix_plot.shape[0]));

        # Set the xlabels
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(160 / corr_matrix_plot.shape[1]));
        plt.title(title, size=14)

    def plot_feature_importances(self, plot_n=15, threshold=None):
        """
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold` cumulative importance.

        Parameters
        --------

        plot_n : int, default = 15
            Number of most important features to plot. Defaults to 15 or the maximum number of features whichever is smaller

        threshold : float, between 0 and 1 default = None
            Threshold for printing information about cumulative importances

        """

        if self.record_zero_importance is None:
            raise NotImplementedError('Feature importances have not been determined. Run `idenfity_zero_importance`')

        # Need to adjust number of features if greater than the features in the data
        if plot_n > self.feature_importances.shape[0]:
            plot_n = self.feature_importances.shape[0] - 1

        self.reset_plot()

        # Make a horizontal bar chart of feature importances
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.feature_importances.index[:plot_n]))),
                self.feature_importances['normalized_importance'][:plot_n],
                align='center', edgecolor='k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(self.feature_importances.index[:plot_n]))))
        ax.set_yticklabels(self.feature_importances['feature'][:plot_n], size=12)

        # Plot labeling
        plt.xlabel('Normalized Importance', size=16);
        plt.title('Feature Importances', size=18)
        plt.show()

        # Cumulative importance plot
        plt.figure(figsize=(6, 4))
        plt.plot(list(range(1, len(self.feature_importances) + 1)), self.feature_importances['cumulative_importance'],
                 'r-')
        plt.xlabel('Number of Features', size=14);
        plt.ylabel('Cumulative Importance', size=14);
        plt.title('Cumulative Feature Importance', size=16);

        if threshold:
            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(self.feature_importances['cumulative_importance'] > threshold))
            plt.vlines(x=importance_index + 1, ymin=0, ymax=1, linestyles='--', colors='blue')
            plt.show();

            print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    def reset_plot(self):
        plt.rcParams = plt.rcParamsDefault

class UserInput:
    """
    The class to contain user input function.

    Returns:
        symbol: stock symbol entered by user

    Raises:
        NameError:: When the symbol user entered is not a valid symbol.
        ValueError: When no or not enough historical data from the source.
    """

    @staticmethod
    def get_symbol():
        """
        This function gets user to enter a stock symbol.
        Exceptions handlers are in place to ensure user enter a valid stock symbol.
        """

        validity = False
        while validity is False:
            try:
                symbol = input("Please enter a NYSE or NASDAQ stock symbol > \b")
                # Make all alphabets uppercase
                symbol = symbol.upper()

                user_confirm = []
                # If user input is not within the expected answers or user just hit enter without entering value
                while user_confirm not in ['n', 'N', 'no', 'No', 'NO', 'y', 'Y', 'yes', 'Yes', 'YES'] and symbol != "":

                    # Get user to confirm his/her input
                    user_confirm = input("Stock quote: [ %s ] is received, enter y/n to confirm >" % symbol)

                    # If user says No
                    if user_confirm in ['n', 'N', 'no', 'No', 'NO']:
                        pass

                    # If user says Yes
                    elif user_confirm in ['y', 'Y', 'yes', 'Yes', 'YES']:
                        print ("Please wait, checking stock symbol's validity ...")
                        try:
                            # Check if data is available for this stock
                            daily_data = pdr.get_data_yahoo(symbol, START, END)
                        except:
                            pass
                        if len(daily_data) > 4700:
                            print ("Great, you have entered a valid stock symbol: {}".format(symbol))
                            validity = True
                        else:
                            validity = False
                            raise ValueError

                    # If user input is not within the expected answers, re-loop and prompt user input again
                    else:
                        pass

            # When stock symbol is not recognized by NASDAQ, chances are it is not a valid stock symbol
            except:
                print('Entry is not a valid stock symbol or not enough of historical data.')
        return symbol

class Data:
    def __init__(self, symbol):
        self.q = symbol
        self._get_daily_data()
        self.technical_indicators_df()

    def _get_daily_data(self):
        """
        This class prepares data by downloading historical data from Yahoo Finance,

        """
        flag = False
        # Set counter for download trial
        counter = 0

        # Safety loop to handle unstable Yahoo finance download
        while not flag and counter < 6:
            try:
                # Define data range
                yf.pdr_override()
                self.daily_data = pdr.get_data_yahoo(self.q, START, END)
                flag = True

            except:
                flag = False
                counter += 1

                if counter < 6:
                    continue
                else:
                    raise Exception("Yahoo finance is down, please try again later. ")

    def technical_indicators_df(self):
        o = self.daily_data['Open'].values
        c = self.daily_data['Close'].values
        h = self.daily_data['High'].values
        l = self.daily_data['Low'].values
        v = self.daily_data['Volume'].astype(float).values
        # define the technical analysis matrix

        # Most data series are normalized by their series' mean
        ta = pd.DataFrame()
        ta['MA5'] = tb.MA(c, timeperiod=5)
        ta['MA10'] = tb.MA(c, timeperiod=10)
        ta['MA20'] = tb.MA(c, timeperiod=20)
        ta['MA60'] = tb.MA(c, timeperiod=60)
        ta['MA120'] = tb.MA(c, timeperiod=120)
        ta['MA5'] = tb.MA(v, timeperiod=5)
        ta['MA10'] = tb.MA(v, timeperiod=10)
        ta['MA20'] = tb.MA(v, timeperiod=20)
        ta['ADX'] = tb.ADX(h, l, c, timeperiod=14)
        ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14)
        ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0]
        ta['RSI'] = tb.RSI(c, timeperiod=14)
        ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
        ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
        ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
        ta['AD'] = tb.AD(h, l, c, v)
        ta['ATR'] = tb.ATR(h, l, c, timeperiod=14)
        ta['HT_DC'] = tb.HT_DCPERIOD(c)
        ta["High/Open"] = h / o
        ta["Low/Open"] = l / o
        ta["Close/Open"] = c / o

        self.ta = ta

    def label(self, df, seq_length):
        return (df['Returns'] > 0).astype(int)

    def scale_data(self):
        """
        This function scale raw data for effective ML training. Stadardization, normalization and MInMax methods are performed.
        """
        # Get the feature and target labels
        self.X_fs = self.X_fs.dropna()

        # Normalization scaling
        self.normalized_scaler = Normalizer()
        normalized = self.normalized_scaler.fit_transform(self.X_fs)
        self._normalized = pd.DataFrame(normalized, index=self.X_fs.index, columns=self.X_fs.columns)

        # Standardization scaling
        self.standardized_scaler = StandardScaler()
        standardized = self.standardized_scaler.fit_transform(self.X_fs)
        self._standardized = pd.DataFrame(standardized, index=self.X_fs.index, columns=self.X_fs.columns)

        # MinMax scaling
        self.minmaxed_scaler = MinMaxScaler(feature_range=(0, 1))
        minmaxed = self.minmaxed_scaler.fit_transform(self.X_fs)
        self._minmaxed = pd.DataFrame(minmaxed, index=self.X_fs.index, columns=self.X_fs.columns)

        self.X_fs = self._minmaxed
        # self.X_fs = self._normalized
        # self.X_fs = self._standardized
        self.y = self.y.loc[self.X_fs.index]

    def preprocessing(self):

        self.daily_data['Returns'] = pd.Series(
            (self.daily_data['Close'] / self.daily_data['Close'].shift(1) - 1) * 100, index=self.daily_data.index)
        seq_length = 3
        self.daily_data['Volume'] = self.daily_data['Volume'].astype(float)
        self.X = self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']]
        self.y = self.label(self.daily_data, seq_length)
        X_shift = [self.X]
        for i in range(1, seq_length):
            shifted_df = self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']].shift(i)
            X_shift.append(shifted_df / shifted_df.mean())
        ohlc = pd.concat(X_shift, axis=1)
        ohlc.columns = sum([[c + 'T-{}'.format(i) for c in ['Open', 'Close', 'High', 'Low', 'Volume']] \
                            for i in range(seq_length)], [])
        self.ta.index = ohlc.index
        self.X = pd.concat([ohlc, self.ta], axis=1)
        self.Xy = pd.concat([self.X, self.y], axis=1)

        fs = FeatureSelector(data=self.X, labels=self.y)
        fs.identify_all(selection_params={'missing_threshold': 0.6,
                                          'correlation_threshold': 0.9,
                                          'task': 'regression',
                                          'eval_metric': 'auc',
                                          'cumulative_importance': 0.99})
        self.X_fs = fs.remove(methods='all', keep_one_hot=True)
        # Add the y label, close price of days ahead(PREDICTION_AHEAD)
        shifted_close_price = self.daily_data['Close'].shift(-PREDICTION_AHEAD)
        self.X_fs['Predict-Y'] = shifted_close_price.loc[self.X_fs.index].dropna()
        self.original = self.X_fs
        self.scale_data()
        self.Xy_fs = pd.concat([self.X_fs, self.y], axis=1)

class Models:

    @staticmethod
    def build_cnn_model(train_X):
        print("\n")
        print("CNN + RNN LSTM model architecture ")
        model = Sequential()
        model.add(Conv1D(activation='linear',
                             kernel_initializer='uniform',
                             bias_initializer='zeros',
                             input_shape=(train_X.shape[1], train_X.shape[2]),
                             #kernel_regularizer=regularizers.l2(0.0001),
                             #activity_regularizer=regularizers.l1(0.0001),
                             filters=256, kernel_size=8))
        model.add(Conv1D(activation='linear',
                             kernel_initializer='uniform',
                             bias_initializer='zeros', filters=256, kernel_size=6))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(3))
        model.add(LSTM(128,
                           kernel_initializer='uniform',
                           bias_initializer='zeros'))
        model.add(Dropout(0.25))
        model.add(Dense(1))
        # optimizer = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # optimizer = keras.optimizers.Adam(lr=0.00001)
        optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0002)
        #optimizer = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0002)
        model.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])
        model.summary()

        return model

    @staticmethod
    def build_rnn_model(train_X):
        # design network
        print("\n")
        print("RNN LSTM model architecture >")
        model = Sequential()
        model.add(LSTM(256, kernel_initializer='random_uniform',
                       bias_initializer='zeros', return_sequences=True,
                       input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.25))
        model.add(LSTM(64, kernel_initializer='random_uniform',
                       # kernel_regularizer=regularizers.l2(0.001),
                       # activity_regularizer=regularizers.l1(0.001),
                       bias_initializer='zeros'))
        model.add(Dropout(0.25))
        model.add(Dense(1))
        optimizer = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0002)
        # optimizer = keras.optimizers.Adagrad(lr=0.03, epsilon=1e-08, decay=0.00002)
        # optimizer = keras.optimizers.Adam(lr=0.0001)
        # optimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
        # optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        # optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

        model.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])
        model.summary()
        return model

class Training:

    def __init__(self, stock_data, stock_fs, stock_context_fs):

        self.stock_data = stock_data
        self.stock_fs = stock_fs
        self.df_values = stock_context_fs.values
        self.stock_context_fs = stock_context_fs

    def plot_training(self, history,nn, symbol):
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.title('Training loss history for {} model for {}'.format(nn, symbol))
        plt.savefig('history_{}_{}.png'.format(nn, symbol))
        plt.show()

    def train_model(self, model, train_X, train_y, model_type, symbol):

        pre_trained = False
        if model_type == "LSTM":
            batch_size = 4
            mc = ModelCheckpoint('best_lstm_model_{}.h5'.format(symbol), monitor='val_loss', save_weights_only=False,
                                 mode='min', verbose=1, save_best_only=True)
            try:
                model = load_model('./best_lstm_model_{}.h5'.format(symbol))
                print("Loading pre-saved model ...")
                pre_trained = True
            except:
                print("No pre-saved model, training new model.")
                pass
        elif model_type == "CNN":
            batch_size = 8
            mc = ModelCheckpoint('best_cnn_model_{}.h5'.format(symbol), monitor='val_loss', save_weights_only=False,
                                 mode='min', verbose=1, save_best_only=True)
            try:
                model = load_model('./best_cnn_model_{}.h5'.format(symbol))
                print("Loading pre-saved model ...")
                pre_trained = True
            except:
                print("No pre-saved model, training new model.")
                pass
        # fit network

        if not pre_trained:
            print("\n")
            print("{} model training starts now for {} ...".format(model_type, symbol))
            print("\n")
            history = model.fit(
                train_X,
                train_y,
                epochs=500,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=2,
                shuffle=True,
                # callbacks=[es, mc, tb, LearningRateTracker()])
                callbacks=[es, mc])

            if model_type == "LSTM":
                model.save('./best_lstm_model_{}.h5'.format(symbol))
            elif model_type == "CNN":
                model.save('./best_cnn_model_{}.h5'.format(symbol))

        elif pre_trained:
            history = []

        return history, model, pre_trained

    def split_data(self, nn):
        # split into train and test sets
        n_train = TRAIN_PORTION * self.df_values.shape[0]
        train = self.df_values[:int(n_train), :]
        test = self.df_values[int(n_train):, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        if nn == "CNN":
            train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
        elif nn == "LSTM":
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        print("\n")
        print("Train feature data shape:", train_X.shape)
        print("Train label data shape:", train_y.shape)
        print("Test feature data shape:", test_X.shape)
        print("Test label data shape:", test_y.shape)

        return train_X, train_y, test_X, test_y

    def scoring(self, model, train_X, train_y, test_X, test_y, nn):

        print("Training and test scores for {} model".format(nn))
        trainScore = model.evaluate(train_X, train_y, verbose=0)
        for i, m in enumerate(model.metrics_names):
            print('Train {0}: {1:5.4f}'.format(m, trainScore[i]))

        testScore = model.evaluate(test_X, test_y, verbose=0)
        for i, m in enumerate(model.metrics_names):
            print('Test {0}: {1:5.4f}'.format(m, testScore[i]))
        print("\n")

    def get_prediction(self, model, train_X, test_X, nn, symbol):

        # Get the predicted price
        predicted_y = model.predict(test_X, batch_size=None, verbose=0, steps=None)
        # Get the trained price
        trained_y = model.predict(train_X, batch_size=None, verbose=0, steps=None)
        # Vertically stack trained and predicted price into a dataframe to form a vector of price produced by CNN
        y = pd.DataFrame(data=np.vstack((trained_y, predicted_y)), columns=[nn], index=self.stock_context_fs.index)
        # Assemble a dataframe with normalized price of original and CNN trained/predicted price
        y_df = pd.concat([self.stock_context_fs[['Predict-Y']], y], axis=1)
        # Assemble the dataframe resembles of the original stock dataframe for inverse transformation.
        df = self.stock_fs.loc[y_df.index]
        # Replace the label column with the CNN trained & predicted price column
        df[['Predict-Y']] = y_df[[nn]]
        # Get it inverse transformed back to normal price
        recovered_data = self.stock_data.minmaxed_scaler.inverse_transform(df)
        recovered_data = pd.DataFrame(data=recovered_data, columns=self.stock_fs.columns, index=df.index)

        Display.plot_prediction(
            self.stock_data.original[['Predict-Y']].loc[recovered_data.index],
            recovered_data[['Predict-Y']], len(trained_y), nn, symbol)

    def modelling(self, nn, symbol):

        train_X, train_y, test_X, test_y = self.split_data(nn)

        if nn == "CNN":
            model = Models.build_cnn_model(train_X)
        elif nn == "LSTM":
            model = Models.build_rnn_model(train_X)
        print("\n")
        history, model, pre_trained = self.train_model(model, train_X, train_y, nn, symbol)
        print("\n")
        if not pre_trained:
            self.plot_training(history, nn, symbol)
        self.scoring(model, train_X, train_y, test_X, test_y, nn)
        self.get_prediction(model, train_X, test_X, nn, symbol)

class Display:

    @staticmethod
    def plot_prediction(original, trained, train_len, nn, symbol):

        """
        Function to plot all portfolio cumulative returns
        """
        # Set a palette so that all 14 lines can be better differentiated
        color_palette = ['#e6194b', '#3cb44b', '#4363d8']
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(original.index, original, '-', label="Original price", linewidth=2, color=color_palette[0])
        ax.plot(trained.iloc[:train_len].index, trained.iloc[:train_len], '-', label="Trained price", linewidth=2,
                color=color_palette[1], alpha=0.8)
        ax.plot(trained.iloc[train_len:].index, trained.iloc[train_len:], '-', label="Predicted price", linewidth=2,
                color=color_palette[2])
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Stock price')
        plt.title('Original, trained & predicted stock price trained on {} model for {}'.format(nn, symbol))
        plt.subplots_adjust(hspace=0.5)

        # Display and save the graph
        plt.savefig('prediction_{}_{}.png'.format(nn, symbol))
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved as prediction_{}.png. When done viewing, please close this plot for next plot. Thank You!".format(
                nn))
        plt.show()

    @staticmethod
    def features_visualization(stock_context_fs, symbol):
        df_values = stock_context_fs.values
        i = 1
        # plot each column
        plt.figure(figsize=[16, 24])
        for i in range(1, len(stock_context_fs.columns)):
            plt.subplot(len(stock_context_fs.columns), 1, i)
            plt.plot(df_values[:, i], lw=1)
            plt.title(stock_context_fs.columns[i], y=0.5, loc='right')
            i += 1
        plt.savefig('features_visualization_{}.png'.format(symbol))
        plt.show()


# ----------------------------- MAIN PROGRAM ---------------------------------
def main():
    """
    The main program

    """
    print("\n")
    print("############################ Price prediction with RNN LSTM & CNN models #################################")
    print("\n")
    # Set the print canvas right
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.width', 1600)

    print("*********************************************  Data Preprocessing ***************************************")
    print("\n")

    symbol = UserInput.get_symbol()
    print("Downloading stock data ...")
    stock_data = Data(symbol)
    market_data = Data("^GSPC")
    print("\n")
    print("Preprocessing and selecting features ...")
    print("\n")
    stock_data.preprocessing()
    stock_raw, stock_fs = stock_data.X, stock_data.X_fs
    market_data.preprocessing()
    market_raw, market_fs = market_data.X, market_data.X_fs

    stock_fs = stock_fs.dropna()
    market_fs.columns = [c + '_M' for c in market_fs.columns]
    market_fs = market_fs.drop(['Predict-Y_M'], axis=1).dropna()
    stock_context_fs = pd.concat([market_fs, stock_fs], axis=1)
    stock_context_fs = stock_context_fs.dropna()

    Display.features_visualization(stock_context_fs, symbol)

    print("Prediction with LSTM RNN:")

    train_model = Training(stock_data, stock_fs, stock_context_fs)
    train_model.modelling("LSTM", symbol)

    print("Prediction with hybrid CNN + RNN:")
    train_model.modelling("CNN", symbol)
    print("\n")

    print ("#######################################   END OF PROGRAM   ###############################################")


if __name__ == '__main__':
    main()

    # -------------------------------- END  ---------------------------------------
