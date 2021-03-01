import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.api as stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint as cointegration
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm import tqdm_notebook as tqdm
from arch import arch_model as ARCH
from scipy import fftpack
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    TimeDistributed,
    Flatten,
)
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
plt.rcParams["font.family"] = "Times New Roman"


def plot_timeseries(timeSeries):
    """ Plots all the Time Series in a given dataframe """

    fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(16, 10))
    fig.suptitle("Cryptocurrency Time Series", size=18)

    for i, ax in enumerate(axes.flatten()):

        data = timeSeries[timeSeries.columns[i]]
        ax.plot(data, color="b", linewidth=1)
        ax.set_title(timeSeries.columns[i], size=16)
        ax.tick_params(labelsize=10)
        ax.grid(True)

    plt.tight_layout()


def plot_graphics(timeSeries):

    """Plots a Boxplot, histogram, QQ-norm plot, Correlogram of all the Time Series
    in the dataframe"""

    for series in timeSeries.columns:

        fig, axes = plt.subplots(
            nrows=3,
            ncols=2,
            dpi=120,
            figsize=(15, 15),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        title = "Graphical Analysis for {}".format(series)
        fig.suptitle(title, size=20)

        axes[0, 0].plot(timeSeries[series], color="b", linewidth=0.8)
        axes[0, 0].set_title("Time Series")
        axes[0, 0].tick_params(labelsize=10)

        sns.distplot(timeSeries[series], kde=True, hist=True, bins=20, ax=axes[0, 1])
        axes[0, 1].set_title("Histogram")
        axes[0, 1].tick_params(labelsize=10)

        plot_acf(timeSeries[series], ax=axes[1, 0])
        axes[1, 0].set_title("Autocorrelation")
        axes[1, 0].tick_params(labelsize=10)

        sns.boxplot(y=timeSeries[series], ax=axes[1, 1], orient="vertical")
        axes[1, 1].set_title("Boxplot")
        axes[1, 1].tick_params(labelsize=10)

        plot_pacf(timeSeries[series], ax=axes[2, 0])
        axes[2, 0].set_title("Partial Autocorrelation")
        axes[2, 0].tick_params(labelsize=10)

        sm.qqplot(timeSeries[series], ax=axes[2, 1], line="q")
        axes[2, 1].set_title("Q-Q Plot against a normal distribution")
        axes[2, 1].tick_params(labelsize=10)

        plt.tight_layout()


def stationary_checking(timeSeries, alpha=0.05):
    """ " This is the implementation of the Augmented Dickey-Fuller
    test which tests the presence of a unit root in the time series provided
    """

    for series in timeSeries.columns:

        print("\n--------------------------------------------\n")
        print("Checking Stationarity of {}".format(series))

        dftest = adfuller(timeSeries[series].dropna(), autolag="AIC", regression="c")

        print("Test Statistic : %.2f, p value : %.5f" % (dftest[0], dftest[1]))

        if dftest[1] <= alpha:
            print("Data is Stationary for {}".format(series))
        else:
            print("Data is NOT Stationary for {}".format(series))


def _normality_checking(series):
    """ "" This functions checks whether ONE time series comes
    from a Normal/Gaussian distribution or not.
    The Null Hypothesis is that the underlying distribution is Gaussian.
    """

    JB_stat, p, _, __ = jarque_bera(series)

    print("\n--------------------------------------------\n")
    print("Checking Normality of {}".format(series.name))
    print("Test Statistic : %.2f, p value : %.5f" % (JB_stat, p))

    alpha = 0.05

    if p > alpha:

        print("Data looks Gaussian: fail to reject the Null Hypothesis")
        return False

    else:

        print("Data does not look Gaussian: we reject the Null Hypothesis")
        return True


def normality_checking(timeSeries):
    """This function checks if all columns are Normally Distributed
    or not by calling the _normality_checking function on each column
    of the dataframe.
    """

    for series in timeSeries:
        _normality_checking(timeSeries[series])


def check_for_cointegration(seriesMain, otherSeries, alpha=0.05, verbose=True):
    """ "" This function checks whether the seriesMain time series
    is co-integrated with the otherSeries time series.
    The Null-hypothesis is that the series are non-cointegrated.
    """

    t_test, p_value, _ = cointegration(seriesMain, otherSeries)

    if p_value < alpha:
        if verbose:
            print("\n--------------------------------\n")
            print("We reject the Null Hypothesis")
            print("The two series are cointegrated")

        return t_test

    else:
        if verbose:
            print("\n--------------------------------\n")
            print("We are unable to reject the Null Hypothesis")
            print("The two series cannot be said to be cointegrated")

        return t_test


# In[18]:


def find_most_cointegrated_pair(allSeries, verbose=True):
    """From given data (allSeries), we all possible ordered pairs of
    time series and find the most co-integrated pair.
    The allSeries should be a dataframe containing the
    datapoints in the rows and the coloumns should be the different time series."""

    for seriesName in allSeries.columns:

        seriesMain = allSeries[seriesName]

        bestTest = 0
        bestPair = {}

        for otherSeries in allSeries.columns.difference([seriesName]):

            t_test = check_for_cointegration(
                seriesMain, allSeries[otherSeries], verbose=verbose
            )

            if abs(t_test) > abs(bestTest):

                bestTest = t_test
                bestPair[seriesName] = otherSeries

    return bestPair, bestTest


# In[19]:


def granger_causality(dataPair, maxlag=10, verbose=False):
    """ " The test checks whether a time series granger-causes another time series.
    The Null Hypothesis is that there exists no Granger causality between the two given series.
    column1 is the resultant series and column2 is the causation series in dataPair
    """

    test = "ssr_ftest"

    res = grangercausalitytests(dataPair, maxlag=maxlag, verbose=verbose)

    bestLag = 0
    fStat = 0

    for lag in res.keys():

        if res[lag][0][test][0] > fStat:

            p_value = res[lag][0][test][1]
            fStat = res[lag][0][test][0]
            bestLag = lag
    print("---------------------------------------------------------------------")
    print(
        "We have achieved a F-Statistic of {:.2f} and p-value of {:.5f} with a Lag-order of {} for Granger Causality".format(
            fStat, p_value, bestLag
        )
    )
    print("---------------------------------------------------------------------")
    return bestLag, fStat


# In[20]:


def find_best_pair_causality(allSeries):
    """ " The function helps us simplify our work by first finding
    the most co-integrated pair and then checking for granger causality
    between the two series.
    """

    bestPair, bestTest = find_most_cointegrated_pair(allSeries, verbose=False)

    series1 = [_ for _ in bestPair.keys()][0]
    series2 = [_ for _ in bestPair.values()][0]

    print("-----------------------------------------------------------------")
    print(
        "{} is most cointegrated with {} for the given timeframe".format(
            series1, series2
        )
    )
    print("-----------------------------------------------------------------")

    dataPair = allSeries[[series1, series2]]

    bestLag, fStat = granger_causality(dataPair, verbose=False)

    return dataPair, bestLag


# In[21]:


def regress(Y, X, verbose=True):

    OLS_results = sm.RLM(Y, X, hasconst=True).fit()

    if verbose:
        print(OLS_results.summary())

    residuals = OLS_results.resid

    residualStatistics = [residuals.mean(), residuals.std()]
    standardResiduals = (residuals - residuals.mean()) / residuals.std()
    confidenceInterval = OLS_results.conf_int()

    return residuals, OLS_results.params, residualStatistics, confidenceInterval


def get_residuals(allSeries):
    """ " This function first finds us the pair that works the best
    for prediction from amongst our data.
    We then regress the causing series onto the prediction series
    to get the set of residuals needed.
    """

    dataPair, bestLag = find_best_pair_causality(allSeries)
    X = dataPair.iloc[:, 1].values.astype(np.float64)[:-bestLag]
    X = stats.add_constant(X)
    Y = dataPair.iloc[:, 0].shift(-bestLag).dropna().values.astype(np.float64)

    residuals, params, residualStatistics, confidenceInterval = regress(Y, X)

    results = {
        "Residuals": residuals,
        "regressParams": params,
        "Residual Statistics": residualStatistics,
        "Best Pair": dataPair.columns,
        "Best Lag": bestLag,
        "Confidence Interval": confidenceInterval,
    }

    return results


def remove_noise_by_fft(series, threshold=0.1):

    frequencies = fftpack.fft(series)
    cleanFreq = []
    for val in range(len(frequencies)):

        if val <= len(frequencies) * threshold:
            cleanFreq.append(frequencies[val])
        else:
            cleanFreq.append(0)

    cleanSeries = fftpack.ifft(cleanFreq)

    return cleanSeries


def loss(regressParams, X, Y):
    def error(y_true, y_pred):
        return custom_loss_function(y_true, y_pred, regressParams, X, Y)

    return error


def custom_loss_function(y_true, y_pred, regressParams, X, Y):

    """" regression constant is a list of [constant,slope]. """

    regressParams = tf.constant(regressParams.astype(np.float32))
    X = tf.constant(X.astype(np.float32))
    Y = tf.constant(Y.astype(np.float32))

    Y_pred = regressParams[0] + regressParams[1] * X + y_pred
    loss = K.mean(K.abs((Y - Y_pred)))

    return loss


def convert_to_window(timeSeries, windowSize, forecastHorizon):
    x = []
    y = []
    for i in range(len(timeSeries) - windowSize - forecastHorizon):

        x.append(timeSeries[i : i + windowSize])
        y.append(timeSeries[i + windowSize + forecastHorizon])

    return np.array(x), np.array(y)


def create_simple_model(
    regressParams, X, Y, WINDOW_SIZE=20, DROP=0.3, NUM_FEATURES=1, BATCH=32
):
    """X is the independent variable here, the stationary, differenced, and standard scaled crypto data
    Y here is the Dependent variable that we are regressing X onto
    """

    model = Sequential(
        [
            LSTM(
                40,
                return_sequences=True,
                recurrent_dropout=0.5,
                input_shape=(WINDOW_SIZE, NUM_FEATURES),
            ),
            Dropout(DROP),
            LSTM(
                20,
                return_sequences=True,
                recurrent_dropout=0.5,
                input_shape=(WINDOW_SIZE, NUM_FEATURES),
            ),
            Dropout(DROP),
            Flatten(),
            Dense(32),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    OPTIMIZER = tf.keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=0.05, nesterov=True)
    LOSS = loss(regressParams=regressParams, X=X, Y=Y)
    model.compile(
        loss=LOSS, optimizer=OPTIMIZER, metrics=["mean_absolute_percentage_error"]
    )
    model.summary()

    return model


def inverse_transform(series, differenced, testSet):
    series = np.add(differenced.std() * series, differenced.mean()) + testSet.shift(1)
    return series


def plot_forecast(predictedArray, algo):

    fig, axes = plt.subplots(dpi=120, figsize=(16, 5))
    axes.fill_between(
        x=predictedArray.index,
        y1=predictedArray.Confidence_Interval_1,
        y2=predictedArray.Confidence_Interval_2,
        alpha=0.15,
        color="red",
    )
    axes.plot(predictedArray.Prediction, color="red")
    axes.plot(predictedArray.Test_Data, color="blue")
    axes.set_title("Forecast of Cryptocurrency by {}".format(algo), size=18)
    axes.set_xlabel("Dates", size=15)
    axes.set_ylabel("Crypto Price", size=15)

    plt.show()


def trading(predictedArray, daily=True, weekly=False):

    initialsum = 10000
    cryptoHoldings = [0]
    portfolio = {"Cash": initialsum, "Crypto Value": 0, "Value": [initialsum]}

    if daily:

        for date in range(len(predictedArray[1:])):

            if predictedArray.Prediction[date + 1] > predictedArray.Test_Data[date]:

                cryptoHoldings.append(1)
                portfolio["Cash"] -= predictedArray.Test_Data[date]
                portfolio["Crypto Value"] = (
                    sum(cryptoHoldings) * predictedArray.Test_Data[date]
                )

            elif predictedArray.Prediction[date + 1] < predictedArray.Test_Data[date]:

                cryptoHoldings.append(-1)
                portfolio["Cash"] += predictedArray.Test_Data[date]
                portfolio["Crypto Value"] = (
                    sum(cryptoHoldings) * predictedArray.Test_Data[date]
                )

            portfolio["Value"].append(portfolio["Cash"] + portfolio["Crypto Value"])

        portfolioHoldings = pd.DataFrame(
            data={
                "Value": portfolio["Value"],
                "Cryptocurrency": np.array(cryptoHoldings).cumsum(),
            },
            index=predictedArray.index,
        )

    if weekly:

        for week in range(0, len(predictedArray[1:]), 7):
            try:

                if predictedArray.Prediction[week + 7] > predictedArray.Test_Data[week]:

                    cryptoHoldings.append(1)
                    portfolio["Cash"] -= predictedArray.Test_Data[week]
                    portfolio["Crypto Value"] = (
                        sum(cryptoHoldings) * predictedArray.Test_Data[week]
                    )

                elif (
                    predictedArray.Prediction[week + 7] < predictedArray.Test_Data[week]
                ):

                    cryptoHoldings.append(-1)
                    portfolio["Cash"] += predictedArray.Test_Data[week]
                    portfolio["Crypto Value"] = (
                        sum(cryptoHoldings) * predictedArray.Test_Data[week]
                    )

                portfolio["Value"].append(portfolio["Cash"] + portfolio["Crypto Value"])

            except:
                continue

        portfolioHoldings = pd.DataFrame(
            data={
                "Value": portfolio["Value"],
                "Cryptocurrency": np.array(cryptoHoldings).cumsum(),
            },
        )

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    fig.suptitle("Trading Strategy for Cryptocurrency", size=18)
    ax[0].plot(portfolioHoldings.Value)
    ax[0].set_title("Value of Portfolio", size=15)
    ax[0].set_ylabel("USD")

    ax[1].plot(portfolioHoldings.Cryptocurrency)
    ax[1].set_title("Cryptoholdings Holdings")
    ax[1].set_ylabel("Number of Crypto")

    return portfolioHoldings


def printt(portfolioHoldings):

    print("-" * 100)
    print(
        "\t \t \t The value of your portfolio is {}".format(
            round(portfolioHoldings.Value.iloc[-1], 6)
        )
    )
    print(
        "\t \t \t You have achieved returns of {}%".format(
            round(
                (
                    (portfolioHoldings.Value.iloc[-1] - portfolioHoldings.Value.iloc[0])
                    / portfolioHoldings.Value.iloc[0]
                )
                * 100,
                2,
            )
        )
    )
    print("-" * 100)
