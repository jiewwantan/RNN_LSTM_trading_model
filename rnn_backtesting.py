# ------------------------- IMPORT LIBRARIES --------------------
import numpy as np
import pandas as pd
import math
from datetime import timedelta
from calendar import isleap
#import h5py
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
import matplotlib.dates as mdates
from matplotlib import gridspec
from datetime import datetime, timedelta
from scipy.stats import norm
from dateutil.relativedelta import relativedelta


# Libraries required by FeatureSelector()
import lightgbm as lgb
import gc
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import copy
from matplotlib.dates import (DateFormatter, WeekdayLocator, DayLocator, MONDAY)

import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
# ------------------------- GLOBAL PARAMETERS -------------------------

# Range of date
START = dt(2000, 1, 1)
END = dt(2017, 2, 11)
END_2 = dt(2018, 12, 15)

START_DATE_RANGE = []
END_DATA_RANGE = []
YRMTH_FMT = mdates.DateFormatter('%b %Y')

TRAIN_PORTION = 0.9
ACCOUNT_FUND = 100000
ALLOCATION_RATIO = 0.1
SINGLE_TRADING_FUND = ACCOUNT_FUND * ALLOCATION_RATIO
# Set price impact for slippage
PRICE_IMPACT = 0.1

# risk free rate, 3-month treasury Yield
RFR = 0.0197
# Dividen
DIV = 0.0
ASSET_N = "Apple Inc"
ASSET = "AAPL"
PCT = 1.0

# ------------------------------ CLASSES ---------------------------------

class Data:
    """
    This class prepares data by downloading historical data from pre-saved data.
    """

    def __init__(self):
        self.load_data()
        self.scale_data()
        self.split_data()

    def load_data(self):
        self.stock_raw_full = pd.read_csv('stock_raw_full.csv', index_col='Date', parse_dates=True,
                                          infer_datetime_format=True)
        self.original_stock_context_fs_full = pd.read_csv('original_stock_context_fs.csv', index_col='Date',
                                                          parse_dates=True, infer_datetime_format=True)
        self.stock_context_fs_full = pd.read_csv('stock_context_fs_full.csv', index_col='Date', parse_dates=True,
                                                 infer_datetime_format=True)
        self.dow_vix = pdr.DataReader('VXDCLS', 'fred', START, END_2, retry_count=10)

    def scale_data(self):
        train_set_size = int(0.9 * len(self.original_stock_context_fs_full))
        train_set = self.original_stock_context_fs_full[:train_set_size]
        test_set = self.original_stock_context_fs_full[train_set_size:]
        # MinMax scaling
        minmaxed_scaler = MinMaxScaler(feature_range=(0, 1))
        self.minmaxed = minmaxed_scaler.fit(train_set)
        train_set_matrix = minmaxed_scaler.transform(train_set)
        test_set_matrix = minmaxed_scaler.transform(test_set)
        train_set_matrix_df = pd.DataFrame(train_set_matrix, index=train_set.index, columns=train_set.columns)
        self.test_set_matrix_df = pd.DataFrame(test_set_matrix, index=test_set.index, columns=test_set.columns)
        self.stock_context_fs_scaled_df = pd.concat([train_set_matrix_df, self.test_set_matrix_df], axis=0)

        print ("Train set shape: ", train_set_matrix.shape)
        print ("Test set shape: ", test_set_matrix.shape)

    def split_data(self):
        df_values = self.stock_context_fs_scaled_df.values
        # split into train and test sets
        n_train = TRAIN_PORTION * df_values.shape[0]
        train = df_values[:int(n_train), :]
        test = df_values[int(n_train):, :]
        # split into input and outputs
        train_X, self.train_y = train[:, :-1], train[:, -1]
        test_X, self.test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        self.test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print("\n")
        print("Train feature data shape:", self.train_X.shape)
        print("Train label data shape:", self.train_y.shape)
        print("Test feature data shape:", self.test_X.shape)
        print("Test label data shape:", self.test_y.shape)

    def get_prediction(self):
        model_lstm = keras.models.load_model('./best_lstm_model_AAPL_strange.h5')
        # Get the predicted price
        self.predicted_y_lstm = model_lstm.predict(self.test_X, batch_size=None, verbose=0, steps=None)
        # Get the trained price
        self.trained_y_lstm = model_lstm.predict(self.train_X, batch_size=None, verbose=0, steps=None)

    def run_prediction(self):
        self.get_prediction()
        # Vertically stack trained and predicted price into a dataframe to form a vector of price produced by CNN
        y_lstm = pd.DataFrame(data=np.vstack((self.trained_y_lstm, self.predicted_y_lstm)), columns=['LSTM'],
                              index=self.stock_context_fs_scaled_df.index)
        # Assemble a dataframe with normalized price of original and CNN trained/predicted price
        lstm_y_df = pd.concat([self.stock_context_fs_scaled_df[['Predict-Y']], y_lstm], axis=1)
        # Assemble the dataframe resembles of the original stock dataframe for inverse transformation.
        lstm_df = self.stock_context_fs_scaled_df.loc[lstm_y_df.index]
        # Replace the label column with the CNN trained & predicted price column
        lstm_df[['Predict-Y']] = lstm_y_df[['LSTM']]
        # Get it inverse transformed back to normal price
        recovered_data_lstm = self.minmaxed.inverse_transform(lstm_df)
        self.recovered_data_lstm = pd.DataFrame(data=recovered_data_lstm, columns=self.stock_context_fs_scaled_df.columns,
                                                index=lstm_df.index)

    def get_train_test_set(self):
        return self.train_X, self.train_y, self.test_X, self.test_y

    def get_all_data(self):
        return self.stock_context_fs_scaled_df, self.stock_raw_full, self.original_stock_context_fs_full, self.stock_context_fs_full, self.dow_vix


class MathCalc:
    """
    This class performs all the mathematical calculations
    """

    def __init__(self, daily_data):

        # Make sure hourly data index is datetime format, not simply string.
        daily_data.index = pd.to_datetime(daily_data.index)
        # Only needs the Close data
        self.daily = daily_data

    @staticmethod
    def diff_year(start_date, end_date):
        """
        This function computes the fractional year for CAGR calculation
        """
        diffyears = end_date.year - start_date.year
        difference = end_date - start_date.replace(end_date.year)
        days_in_year = isleap(end_date.year) and 366 or 365
        difference_in_years = diffyears + (difference.days + difference.seconds / 86400.0) / days_in_year
        return difference_in_years

    @staticmethod
    def cagr(portfolio_value):
        """
        This function computes CAGR
        """
        st = portfolio_value.index[0]
        en = portfolio_value.index[-1]
        num_year = MathCalc.diff_year(st, en)
        return (portfolio_value[-1] / portfolio_value[0]) ** (1.0 / float(num_year)) - 1

    @staticmethod
    def calc_return(period):
        """
        This function compute the return of a series
        """
        period_return = period / period.shift(1) - 1
        return period_return[1:len(period_return)]

    @staticmethod
    def max_drawdown(r):
        """
        This function calculates maximum drawdown occurs in a series of cummulative returns
        """
        dd = r.div(r.cummax()).sub(1)
        maxdd = dd.min()
        return round(maxdd, 2)

    @staticmethod
    def calc_gain_to_pain(returns):
        """
        This function computes the gain to pain ratio given a series of profits and losses

        """
        profit_loss = np.array(returns)
        sum_returns = returns.sum()
        sum_neg_months = abs(returns[returns < 0].sum())
        gain_to_pain = sum_returns / sum_neg_months

        # print "Gain to Pain ratio: ", gain_to_pain
        return gain_to_pain

    @staticmethod
    def calc_lake_ratio(series):

        """
        This function computes lake ratio

        """
        water = 0
        earth = 0
        series = series.dropna()
        water_level = []
        for i, s in enumerate(series):
            if i == 0:
                peak = s
            else:
                peak = np.max(series[0:i])
            water_level.append(peak)
            if s < peak:
                water = water + peak - s
            earth = earth + s
        return water / earth

    @staticmethod
    def construct_book(stocks_values):
        """
        This function construct the trading book for stock trading
        """
        portfolio = pd.DataFrame(index=stocks_values.index,
                                 columns=["Total Values", "ProfitLoss", "Returns", "CumReturns"])
        portfolio["Total Values"] = stocks_values
        portfolio["ProfitLoss"] = portfolio["Total Values"] - portfolio["Total Values"].shift(1).fillna(
            portfolio["Total Values"][0])
        portfolio["Returns"] = portfolio["Total Values"] / portfolio["Total Values"].shift(1) - 1
        portfolio["CumReturns"] = portfolio["Returns"].add(1).cumprod().fillna(1)

        return portfolio

    @staticmethod
    def winpct(realized_pnl):
        return float(len(realized_pnl[realized_pnl > 0])) / float(len(realized_pnl)) * 100

    @staticmethod
    def winloss(realized_pnl):
        """
        This function calculates win to loss ratio
        """
        return float(len(realized_pnl[realized_pnl > 0])) / float(len(realized_pnl[realized_pnl < 0]))

    @staticmethod
    def meanreturn_trade(realized_pnl, current_value):
        """
        This function calculates the mean of all trade returns
        """

        previous_value = current_value - realized_pnl
        trade_return = realized_pnl / previous_value

        return trade_return.mean()

    @staticmethod
    def longestconsecutive_loss(arr):
        """
        This function computes the longest losing streak
        """

        # remove all non trading activities
        arr = list(filter(lambda a: a != 0, arr))

        n = len(arr)
        # Initialize result
        res = 0

        # Traverse array
        for i in range(n):

            # Count of current
            # non-negative integers
            curr_count = 0
            while (i < n and arr[i] < 0):
                curr_count += 1
                i += 1

            # Update result if required.
            res = max(res, curr_count)

        return res

    @staticmethod
    def calc_kpi(portfolio, stock_values, symbol):
        """
        This function calculates individual portfolio KPI related its risk-return profile
        """
        KPI = ['Win %', 'Win to Loss Ratio', 'Max Consecutive Losers', 'Max dd', 'CAGR',
               'Lake ratio', 'Gain to Pain']

        kpi = pd.DataFrame(index=[symbol], columns=KPI)
        try:
            kpi['Win %'] = MathCalc.winpct(stock_values["Profit & Loss"])
            kpi['Win to Loss Ratio'] = MathCalc.winloss(stock_values["Profit & Loss"])
            kpi['Max Consecutive Losers'] = MathCalc.longestconsecutive_loss(stock_values["Profit & Loss"])
            kpi['CAGR'].iloc[0] = MathCalc.cagr(portfolio["Total Values"])
            kpi['Max dd'].iloc[0] = MathCalc.max_drawdown(portfolio["CumReturns"])
            kpi['Lake ratio'].iloc[0] = MathCalc.calc_lake_ratio(portfolio['CumReturns'])
            kpi['Gain to Pain'].iloc[0] = MathCalc.calc_gain_to_pain(portfolio['Returns'])
        except:
            kpi['Win %'] = float('nan')
            kpi['Win to Loss Ratio'] = float('nan')
            kpi['Max Consecutive Losers'] = float('nan')
            kpi['CAGR'].iloc[0] = MathCalc.cagr(stock_values)
            kpi['Max dd'].iloc[0] = MathCalc.max_drawdown(portfolio["CumReturns"])
            kpi['Lake ratio'].iloc[0] = MathCalc.calc_lake_ratio(portfolio['CumReturns'])
            kpi['Gain to Pain'].iloc[0] = MathCalc.calc_gain_to_pain(portfolio['Returns'])

        return kpi


class UserInterfaceDisplay:
    """
    The class to display plot(s) to users
    """

    def __init__(self, symbol):
        self.symbol = symbol

    def plot_signal(self, series, signal):
        """
        This function plots the time series together with respective trading signals and indicators

        """

        month = mdates.AutoDateLocator()
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1])
        ax1 = plt.subplot(gs[0])
        ax1.set_title("{} price evolution and RNN LSTM model generated trade signals".format(self.symbol),
                      fontsize=15)
        ax1.plot(series.index, series, label='{}'.format(self.symbol), c='#ff4811', linewidth=2)
        ax1.set_ylabel(self.symbol)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.8, ls=':')
        ax1.xaxis.set_major_locator(month)
        ax1.xaxis.set_major_formatter(YRMTH_FMT)
        ax1.xaxis_date()
        ax1.autoscale_view()
        plt.setp(plt.gca().get_xticklabels(), rotation=0)

        ax2 = plt.subplot(gs[1])
        ax2.plot(signal.index, signal, linewidth=0.5)
        ax2.set_ylabel('Trade signal')
        ax2.xaxis.set_major_locator(month)
        ax2.xaxis.set_major_formatter(YRMTH_FMT)
        ax2.grid(True, alpha=0.8, ls=':')
        plt.setp(plt.gca().get_xticklabels(), rotation=0)

        print("Plot saved as {}_trade_signal.png. Please close this plot for next plot".format(self.symbol))
        plt.savefig('{}_trade_signal.png'.format(self.symbol))
        plt.show()

    def plot_returns(self, cum_returns_model, cum_returns_buyhold):
        """
        Function to plot the trade cumulative returns
        """
        trading_days = cum_returns_model.index[-1] - cum_returns_model.index[0]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cum_returns_model.index, cum_returns_model, '-',
                label="RNN + delta hedged model trading for {}".format(self.symbol), linewidth=2.5, color='g')
        ax.plot(cum_returns_buyhold.index, cum_returns_buyhold, '-',
                label="Buy and hold trading for {}".format(self.symbol), linewidth=2.5, color='b')
        plt.legend()
        plt.xlabel('Trading timeline')
        plt.ylabel('Cumulative returns')
        plt.title('{} days cumulative returns for {}'.format(trading_days.days, self.symbol))
        # Display and save the graph
        plt.savefig('{}_cumreturns.png'.format(self.symbol))
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved as {}_cumreturns.png. When done viewing, please close this plot. Thank You!".format(
                self.symbol))
        plt.show()

    def plot_deviation(self, prediction_deviation):

        color_palette = ['#4363d8']
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(prediction_deviation.index, prediction_deviation, '-', label="Prediction deviation", linewidth=2,
                color=color_palette[0])
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Deviation from label')
        plt.title('Prediction deviation for stock price trained on RNN model for {}'.format(ASSET))
        plt.subplots_adjust(hspace=0.5)

        # Display and save the graph
        plt.savefig('prediction_deviation_{}.png'.format(ASSET))
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved as prediction_deviation_{}.png. When done viewing, please close this plot for next plot. Thank You!".format(
                ASSET))
        plt.show()

    def plot_prediction(self, original, trained, train_len, nn):
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
        plt.title('Original, trained & predicted stock price trained on {} model for {}'.format(nn, ASSET))
        plt.subplots_adjust(hspace=0.5)

        # Display and save the graph
        plt.savefig('prediction_{}_{}.png'.format(nn, ASSET))
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved as prediction_{}.png. When done viewing, please close this plot for next plot. Thank You!".format(
                nn))
        plt.show()



class Option:
    """
    This class computes option greeks & premiums using Black-scholes calculations
    """

    def __init__(self, right, s, k, eval_date, exp_date, price=None, rf=0.01, vol=0.3,
                 div=0):
        self.k = float(k)
        self.s = float(s)
        self.rf = float(rf)
        self.vol = float(vol)
        self.eval_date = eval_date
        self.exp_date = exp_date
        self.t = self.calculate_t()
        if self.t == 0: self.t = 0.000001  ## Case valuation in expiration date
        self.price = price
        self.right = right  ## 'C' or 'P'
        self.div = div

    def calculate_t(self):
        if isinstance(self.eval_date, str):
            if '/' in self.eval_date:
                (day, month, year) = self.eval_date.split('/')
            else:
                (day, month, year) = self.eval_date[6:8], self.eval_date[4:6], self.eval_date[0:4]
            d0 = datetime(int(year), int(month), int(day))
        elif type(self.eval_date) == float or type(self.eval_date) == long or type(self.eval_date) == np.float64:
            (day, month, year) = (str(self.eval_date)[6:8], str(self.eval_date)[4:6], str(self.eval_date)[0:4])
            d0 = datetime(int(year), int(month), int(day))
        else:
            d0 = self.eval_date

        if isinstance(self.exp_date, str):
            if '/' in self.exp_date:
                (day, month, year) = self.exp_date.split('/')
            else:
                (day, month, year) = self.exp_date[6:8], self.exp_date[4:6], self.exp_date[0:4]
            d1 = datetime(int(year), int(month), int(day))
        elif type(self.exp_date) == float or type(self.exp_date) == long or type(self.exp_date) == np.float64:
            (day, month, year) = (str(self.exp_date)[6:8], str(self.exp_date)[4:6], str(self.exp_date)[0:4])
            d1 = datetime(int(year), int(month), int(day))
        else:
            d1 = self.exp_date

        return (d1 - d0).days / 365.0

    def get_price_delta(self):
        d1 = (math.log(self.s / float(self.k)) + (self.rf + self.div + math.pow(self.vol, 2) / 2.0) * self.t) / float(
            self.vol * math.sqrt(self.t))
        d2 = d1 - self.vol * math.sqrt(self.t)
        if self.right == 'C':
            self.calc_price = (norm.cdf(d1) * self.s * math.exp(-self.div * self.t) - norm.cdf(d2) * self.k * math.exp(
                -self.rf * self.t))
            self.delta = norm.cdf(d1)
        elif self.right == 'P':
            self.calc_price = (
                    -norm.cdf(-d1) * self.s * math.exp(-self.div * self.t) + norm.cdf(-d2) * self.k * math.exp(
                -self.rf * self.t))
            self.delta = -norm.cdf(-d1)

    def get_call(self):
        d1 = (math.log(self.s / self.k) + (self.rf + math.pow(self.vol, 2) / 2.0) * self.t) / (
                self.vol * math.sqrt(self.t))
        d2 = d1 - self.vol * math.sqrt(self.t)
        self.call = (norm.cdf(d1) * self.s - norm.cdf(d2) * self.k * math.exp(-self.rf * self.t))
        # put =  ( -norm.cdf(-d1) * self.s + norm.cdf(-d2) * self.k * math.exp( -self.rf * self.t ) )
        self.call_delta = norm.cdf(d1)

    def get_put(self):
        d1 = (math.log(self.s / self.k) + (self.rf + math.pow(self.vol, 2) / 2) * self.t) / (
                self.vol * math.sqrt(self.t))
        d2 = d1 - self.vol * math.sqrt(self.t)
        # call = ( norm.cdf(d1) * self.s - norm.cdf(d2) * self.k * math.exp( -self.rf * self.t ) )
        self.put = (-norm.cdf(-d1) * self.s + norm.cdf(-d2) * self.k * math.exp(-self.rf * self.t))
        self.put_delta = -norm.cdf(-d1)

    def get_theta(self, dt=0.0027777):
        self.t += dt
        self.get_price_delta()
        after_price = self.calc_price
        self.t -= dt
        self.get_price_delta()
        orig_price = self.calc_price
        self.theta = (after_price - orig_price) * (-1)

    def get_gamma(self, ds=0.01):
        self.s += ds
        self.get_price_delta()
        after_delta = self.delta
        self.s -= ds
        self.get_price_delta()
        orig_delta = self.delta
        self.gamma = (after_delta - orig_delta) / ds

    def get_all(self):
        self.get_price_delta()
        self.get_theta()
        self.get_gamma()
        return self.calc_price, self.delta, self.theta, self.gamma

    def get_impl_vol(self):
        """
        This function will iterate until finding the implied volatility
        """
        ITERATIONS = 100
        ACCURACY = 0.05
        low_vol = 0
        high_vol = 1
        self.vol = 0.5
        self.get_price_delta()
        for i in range(ITERATIONS):
            if self.calc_price > self.price + ACCURACY:
                high_vol = self.vol
            elif self.calc_price < self.price - ACCURACY:
                low_vol = self.vol
            else:
                break
            self.vol = low_vol + (high_vol - low_vol) / 2.0
            self.get_price_delta()

        return self.vol


class Trading:

    def __init__(self, recovered_data_lstm, stock_raw_full, test_set, dow_vix):
        self.daily_c = stock_raw_full.loc[test_set.index]['CloseT-0']
        self.daily_v = stock_raw_full.loc[test_set.index]['VolumeT-0']
        self.test_set = test_set
        self.dow_vix = dow_vix

        self.generate_signals(recovered_data_lstm)

    def slippage_price(self, order, price, stock_quantity, day_volume):
        """
        This function performs slippage price calculation using Zipline's volume share model
        https://www.zipline.io/_modules/zipline/finance/slippage.html
        """

        volumeShare = stock_quantity / float(day_volume)
        impactPct = volumeShare ** 2 * PRICE_IMPACT

        if order > 0:
            slipped_price = price * (1 + impactPct)
        else:
            slipped_price = price * (1 - impactPct)

        # print order, " price: ", price, "slipped price: ", slipped_price
        return slipped_price

    def commission(self, num_share, share_value):
        """
        This function computes commission fee of every trade
        https://www.interactivebrokers.com/en/index.php?f=1590&p=stocks1
        """

        comm_fee = 0.005 * num_share
        max_comm_fee = 0.005 * share_value

        if num_share < 1.0:
            comm_fee = 1.0
        elif comm_fee > max_comm_fee:
            comm_fee = max_comm_fee

        return comm_fee

    def calc_option_premium(self, day):

        # Just in case DOW VIX is not available on the stock trading day
        while day not in self.dow_vix.index:
            day = day - timedelta(days=1)
        # Put option
        right = 'P'
        # Proxy volatility with Cboe DJIA Volatility Index, VXD

        vol = self.dow_vix.loc[day].VXDCLS / 100
        # Current underlying price
        s = self.daily_c.loc[day]
        # Strike price is 3 strikes out-of-money, +3.5 is approximately 3-4 strikes away
        k = round(s, 0) + 0
        # Current date when option is transacted
        eval_date = day.strftime('%Y%m%d')
        # Expiry date
        exp_date = (day + relativedelta(weeks=+2)).strftime('%Y%m%d')
        opt_contract = Option(s=s, k=k, eval_date=eval_date, exp_date=exp_date, rf=RFR, vol=vol, right=right,
                              div=DIV)
        premium, delta, theta, gamma = opt_contract.get_all()
        return premium

    def generate_signals(self, recovered_data_lstm):

        predicted_tomorrow_close = recovered_data_lstm.loc[self.test_set.index]['Predict-Y']
        today_close = self.daily_c.loc[predicted_tomorrow_close.index]
        predicted_next_day_returns = predicted_tomorrow_close / predicted_tomorrow_close.shift(1) - 1
        next_day_returns = today_close / today_close.shift(1) - 1
        signals = pd.DataFrame(index=predicted_tomorrow_close.index, columns=["Signal"])
        for d in predicted_tomorrow_close.index:
            if predicted_tomorrow_close.loc[d] > today_close.loc[d] and next_day_returns.loc[d] > 0 and \
                            predicted_next_day_returns.loc[d] > 0:
                signals.loc[d]["Signal"] = 2
            elif predicted_tomorrow_close.loc[d] < today_close.loc[d] and next_day_returns.loc[d] < 0 and \
                            predicted_next_day_returns.loc[d] < 0:
                signals.loc[d]["Signal"] = -2
            elif predicted_tomorrow_close.loc[d] > today_close.loc[d]:
                signals.loc[d]["Signal"] = 2
            elif next_day_returns.loc[d] > 0:
                signals.loc[d]["Signal"] = 1
            elif next_day_returns.loc[d] < 0:
                signals.loc[d]["Signal"] = -1
            elif predicted_next_day_returns.loc[d] > 0:
                signals.loc[d]["Signal"] = 2
            elif predicted_next_day_returns.loc[d] < 0:
                signals.loc[d]["Signal"] = -1
            else:
                signals.loc[d]["Signal"] = 0
        self.signals = signals

    def execute_trading(self):
        """
        This function performs long only trades.
        """
        # Call up trading signla caculation
        account_value = ACCOUNT_FUND
        stocks_values = pd.DataFrame(index=self.daily_c.index,
                                     columns=["Stock Price", "Stock Quantity", "Options Quantity",
                                              "Profit & Loss", "Trade Returns",
                                              "Portfolio Value", "Options Value", "Account Value",
                                              "Total Value"])
        stock_quantity = 0
        account_profit_holder = 0
        account_equity_holder = 0
        contract_to_hedge = 0
        premium_value = 0

        # Slide through the timeline
        for d in self.daily_c.index:
            # if this is the first hour and signal is buy
            if (d == self.daily_c.index[0]) and (stock_quantity == 0) and (self.signals.loc[d]['Signal'] >= 1):
                if self.signals.loc[d]['Signal'] == 1:
                    stock_quantity = SINGLE_TRADING_FUND / self.daily_c.loc[d]
                    portfolio_value = SINGLE_TRADING_FUND
                elif self.signals.loc[d]['Signal'] == 2:
                    stock_quantity = SINGLE_TRADING_FUND * 2.0 / self.daily_c.loc[d]
                    portfolio_value = SINGLE_TRADING_FUND * 2.0
                slipped_price = self.slippage_price(self.signals.loc[d]['Signal'], self.daily_c.loc[d],
                                                    stock_quantity,
                                                    self.daily_v.loc[d])

                # 1 contract is equivalent to 100 delta which covers 100 stocks, calculate the contract to delta hedge
                # assuming fractional contract is avialable if stock_quantity is less than is not an integer number

                options_premium = self.calc_option_premium(d)
                contract_to_hedge = stock_quantity / 100 * PCT
                premium_value = contract_to_hedge * options_premium * 100

                realized_pnl = 0.0
                realized_ret = float('nan')
                buy_price = slipped_price
                commission_cost = self.commission(stock_quantity, portfolio_value)
                account_value = account_value - portfolio_value - commission_cost - premium_value

            # if this the first hour and no trading signal
            elif d == self.daily_c.index[0] and self.signals.loc[d]['Signal'] < 1:
                stock_quantity = 0
                portfolio_value = 1
                realized_pnl = 0.0
                realized_ret = float('nan')
                buy_position = 0
                premium_value = 0

            # if there's existing position and trading signal is sell
            elif stock_quantity > 0 and self.signals.loc[d]['Signal'] < 0:
                slipped_price = self.slippage_price(self.signals.loc[d]['Signal'], self.daily_c.loc[d],
                                                    stock_quantity,
                                                    self.daily_v.loc[d])

                # Close the hedge also
                options_premium = self.calc_option_premium(d)
                premium_value = contract_to_hedge * options_premium * 100

                realized_pnl = stock_quantity * (slipped_price - buy_price) + premium_value

                realized_ret = realized_pnl / (stock_quantity * buy_price)
                commission_cost = self.commission(stock_quantity, (stock_quantity * slipped_price))
                account_value = account_value + (
                stock_quantity * slipped_price) - commission_cost + premium_value
                stock_quantity = 0
                portfolio_value = 0.0
                premium_value = 0
                contract_to_hedge = 0


            # With position, hold and no trading signal, just update portfolio value with latest price
            elif stock_quantity > 0 and self.signals.loc[d]['Signal'] >= 0:
                portfolio_value = stock_quantity * self.daily_c.loc[d]
                realized_pnl = 0.0
                realized_ret = float('nan')

                options_premium = self.calc_option_premium(d)
                premium_value = contract_to_hedge * options_premium * 100

            # With no position, trading signal is buy
            elif stock_quantity == 0 and self.signals.loc[d]['Signal'] >= 1:
                if self.signals.loc[d]['Signal'] == 1:
                    stock_quantity = SINGLE_TRADING_FUND / self.daily_c.loc[d]
                    portfolio_value = SINGLE_TRADING_FUND
                elif self.signals.loc[d]['Signal'] == 2:
                    stock_quantity = SINGLE_TRADING_FUND * 2 / self.daily_c.loc[d]
                    portfolio_value = SINGLE_TRADING_FUND * 2
                slipped_price = self.slippage_price(self.signals.loc[d]['Signal'], self.daily_c.loc[d],
                                                    stock_quantity,
                                                    self.daily_v.loc[d])
                buy_price = slipped_price
                realized_pnl = 0.0
                realized_ret = float('nan')
                commission_cost = self.commission(stock_quantity, slipped_price * stock_quantity)
                options_premium = self.calc_option_premium(d)
                contract_to_hedge = stock_quantity / 100 * PCT
                premium_value = contract_to_hedge * options_premium * 100

                account_value = account_value - (
                slipped_price * stock_quantity) - commission_cost - premium_value


            # With no position, trading signal is not buy, do nothing
            elif stock_quantity == 0 and self.signals.loc[d]['Signal'] < 1:
                realized_pnl = 0.0
                realized_ret = float('nan')
                premium_value = 0

            # Record it in the stock position value book
            stocks_values["Profit & Loss"].loc[d] = realized_pnl
            stocks_values["Trade Returns"].loc[d] = realized_ret
            stocks_values["Stock Quantity"].loc[d] = stock_quantity
            stocks_values["Options Quantity"].loc[d] = contract_to_hedge
            stocks_values["Portfolio Value"].loc[d] = portfolio_value
            stocks_values["Options Value"].loc[d] = premium_value
            stocks_values["Stock Price"].loc[d] = self.daily_c.loc[d]
            stocks_values["Account Value"].loc[d] = account_value
            account_equity = stocks_values["Portfolio Value"].loc[d] + stocks_values["Account Value"].loc[d]
            account_profit = stocks_values["Profit & Loss"].sum()

        stocks_values["Total Value"] = stocks_values["Portfolio Value"] + stocks_values["Account Value"] + \
                                       stocks_values["Options Value"]
        # Calculate trading book
        portfolio_returns = MathCalc.construct_book(stocks_values["Total Value"])
        # Calculate trade KPI
        kpi = MathCalc.calc_kpi(portfolio_returns, stocks_values, ASSET)
        return portfolio_returns, kpi, stocks_values, self.signals

    def buyandhold_trade(self):
        """
        This function performs a long only trade on 10 randomly chosen Dow stocks on the first day of trading, hold the
        stocks until the last trading day in the window.
        """
        # Calculate equally weighted fund allocation for each stock

        stock_quantity = SINGLE_TRADING_FUND / self.daily_c.iloc[0]
        stocks_values = self.daily_c.mul(stock_quantity) + (ACCOUNT_FUND - SINGLE_TRADING_FUND)
        portfolio_returns = MathCalc.construct_book(stocks_values)
        kpi = MathCalc.calc_kpi(portfolio_returns, stocks_values, ASSET)

        return portfolio_returns, kpi, stocks_values
    # ----------------------------- MAIN PROGRAM ---------------------------------


def main():
    """
    The main program

    """
    print ("\n")
    print ("############################ 20 months of prediction with RNN model trained with 17-year data   #################################")
    print ("\n")
    # Set the print canvas right
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1600)

    # Get the necessary data from user's choice of stock symbol
    data_processing = Data()
    stock_context_fs_scaled_df, stock_raw_full, original_stock_context_fs_full, stock_context_fs_full, dow_vix = data_processing.get_all_data()
    print ("\n")
    print ("With all required model and data loaded and downloaded, prediction starts .. ")
    print ("\n")
    data_processing.run_prediction()
    recovered_data_lstm = data_processing.recovered_data_lstm
    train_len = len(data_processing.trained_y_lstm)
    test_set = data_processing.test_set_matrix_df

    trading_plot = UserInterfaceDisplay(ASSET)

    trading_plot.plot_prediction(
        original_stock_context_fs_full[['Predict-Y']].loc[recovered_data_lstm.index],
        recovered_data_lstm[['Predict-Y']], train_len, "LSTM")

    prediction_deviation = recovered_data_lstm.loc[test_set.index][['Predict-Y']] - \
                           original_stock_context_fs_full.loc[test_set.index][['Predict-Y']]

    trading_plot.plot_deviation(prediction_deviation)
    stock_trading = Trading(recovered_data_lstm, stock_raw_full, test_set, dow_vix)

    portfolio_returns_model, kpi_model, stocks_values_model, signals_model = stock_trading.execute_trading()
    portfolio_returns_buyhold, kpi_buyhold, stocks_values_buyhold = stock_trading.buyandhold_trade()

    print ("\n")
    print(kpi_model)
    print ("\n")
    print(kpi_buyhold)
    print ("\n")


    trading_plot.plot_signal(stock_raw_full.loc[signals_model.index][['CloseT-0']], signals_model)
    trading_plot.plot_returns(portfolio_returns_model[['CumReturns']], portfolio_returns_buyhold[['CumReturns']])
    print ("#######################################   END OF PROGRAM   ###############################################")


if __name__ == '__main__':
    main()

    # -------------------------------- END  ---------------------------------------
