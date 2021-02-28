

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as stats
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint as cointegration
import warnings
from arch import arch_model as ARCH
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm import tqdm_notebook as tqdm
import statsmodels.api as sm
import seaborn as sns
from scipy import fftpack
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
plt.rcParams["font.family"] = "Times New Roman"


# In[2]:


data = pd.read_csv(r'data\daily\gemini_LTCUSD_day.csv',index_col='Date')
data.dropna(inplace=True)
index=[]

for date in data.index:

    index.append(dt.datetime.strptime(date,"%d/%m/%Y %H:%M"))

data.index=index
data.sort_index(inplace=True)
data.columns = ['Litecoin','Zcash','Ethereum','Bitcoin']

crypto = data.copy()


# In[3]:


def plot_timeseries(timeSeries):
    """ Plots all the Time Series in a given dataframe """
    
    fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(16,5))
    fig.suptitle('Cryptocurrency Time Series', size=18)
    
    for i, ax in enumerate(axes.flatten()):
    
        data = crypto[timeSeries.columns[i]]
        ax.plot(data, color='b', linewidth=1)
        ax.set_title(crypto.columns[i],size=16)
        ax.tick_params(labelsize=10)
        ax.grid(True)

    plt.tight_layout();

    


# In[4]:


plot_timeseries(timeSeries=crypto)


# In[5]:


def plot_graphics(timeSeries):
    
    """ Plots a Boxplot, histogram, QQ-norm plot, Correlogram of all the Time Series
    in the dataframe """

    
    for series in timeSeries.columns:
        
        fig, axes = plt.subplots(nrows=3, ncols=2,
                                 dpi=180, figsize=(15,15),
                                 gridspec_kw={'width_ratios': [2, 1]})
        title = "Graphical Analysis for {}".format(series)
        fig.suptitle(title, size=20)
        
        axes[0,0].plot( timeSeries[series],
                        color='b', linewidth=0.8)
        axes[0,0].set_title('Time Series')
        axes[0,0].tick_params(labelsize=10)
        
        sns.distplot(timeSeries[series], kde=True,
                     hist=True, bins=20, ax=axes[0,1])
        axes[0,1].set_title("Histogram")
        axes[0,1].tick_params(labelsize=10)
        
        
        plot_acf(timeSeries[series], ax=axes[1,0])
        axes[1,0].set_title("Autocorrelation")
        axes[1,0].tick_params(labelsize=10)
        
        sns.boxplot(y=timeSeries[series], ax=axes[1,1],
                    orient="vertical")
        axes[1,1].set_title("Boxplot")
        axes[1,1].tick_params(labelsize=10)
        
        plot_pacf(timeSeries[series], ax=axes[2,0])
        axes[2,0].set_title("Partial Autocorrelation")
        axes[2,0].tick_params(labelsize=10)        
        
        sm.qqplot(timeSeries[series], ax=axes[2,1],
                  line="q")
        axes[2,1].set_title("Q-Q Plot against a normal distribution")
        axes[2,1].tick_params(labelsize=10)

        plt.tight_layout();


# In[6]:


plot_graphics(crypto)


# In[7]:


def _normality_checking(series):
    """"" This functions checks whether ONE time series comes
    from a Normal/Gaussian distribution or not.
    The Null Hypothesis is that the underlying distribution is Gaussian.
    """
    
    JB_stat, p,_,__ = jarque_bera(series)
    
    print('\n--------------------------------------------\n')
    print('Checking Normality of {}'.format(series.name))
    print('Test Statistic : %.2f, p value : %.5f' % (JB_stat, p))
    
    alpha = 0.05
    
    if p > alpha:
        
        print('Data looks Gaussian: fail to reject the Null Hypothesis')
        return False
    
    else:
        
        print('Data does not look Gaussian: we reject the Null Hypothesis')
        return True


# In[8]:


def normality_checking(timeSeries):
    """ This function checks if all columns are Normally Distributed
    or not by calling the _normality_checking function on each column 
    of the dataframe. 
    """
    
    for series in timeSeries:
        _normality_checking(timeSeries[series])


# In[9]:


normality_checking(crypto)


# In[10]:


def stationary_checking(timeSeries,alpha=0.05):
    """" This is the implementation of the Augmented Dickey-Fuller
    test which tests the presence of a unit root in the time series provided
    """
    
    for series in timeSeries.columns:
        
        print('\n--------------------------------------------\n')
        print('Checking Stationarity of {}'.format(series))
        
        dftest=adfuller(timeSeries[series].dropna(),
                        autolag='AIC', regression='c')
        
        print('Test Statistic : %.2f, p value : %.5f' % (dftest[0], dftest[1]))

        if dftest[1]<=alpha:
            print("Data is Stationary for {}".format(series))
        else:
            print("Data is NOT Stationary for {}".format(series))
            


# In[11]:


stationary_checking(crypto)


# In[12]:


differencedCrypto = crypto.diff().bfill()
standardDifferencedCrypto = (differencedCrypto - differencedCrypto.mean()) / differencedCrypto.std()


# In[13]:


plot_graphics(standardDifferencedCrypto)


# In[14]:


normality_checking(standardDifferencedCrypto)


# In[15]:


stationary_checking(standardDifferencedCrypto)


# In[17]:


def check_for_cointegration(seriesMain,otherSeries,alpha=0.05,verbose = True):
    """"" This function checks whether the seriesMain time series
    is co-integrated with the otherSeries time series.
    The Null-hypothesis is that the series are non-cointegrated.
    """
    
    t_test, p_value, _ = cointegration(seriesMain,otherSeries)
    
    
    if p_value < alpha:
        if verbose:
            print('\n--------------------------------\n')
            print("We reject the Null Hypothesis")
            print('The two series are cointegrated')
            
        return t_test
    
    else: 
        if verbose:
            print('\n--------------------------------\n')
            print("We are unable to reject the Null Hypothesis")
            print("The two series cannot be said to be cointegrated")
        
        return t_test  


# In[18]:


def find_most_cointegrated_pair(allSeries,verbose=True):
    """ From given data (allSeries), we all possible ordered pairs of
    time series and find the most co-integrated pair.
    The allSeries should be a dataframe containing the
    datapoints in the rows and the coloumns should be the different time series."""
    
    for seriesName in allSeries.columns:
        
        seriesMain = allSeries[seriesName]
        
        bestTest = 0
        bestPair = {}
        
        for otherSeries in allSeries.columns.difference([seriesName]):
            
            t_test = check_for_cointegration(seriesMain,
                                             allSeries[otherSeries],
                                             verbose=verbose)
            
            if abs(t_test) > abs(bestTest):
                
                bestTest = t_test
                bestPair[seriesName] = otherSeries
                
    return bestPair, bestTest    


# In[19]:


def granger_causality(dataPair,maxlag=10,verbose=False):
    """" The test checks whether a time series granger-causes another time series. 
    The Null Hypothesis is that there exists no Granger causality between the two given series.
    column1 is the resultant series and column2 is the causation series in dataPair
    """
    
    test = 'ssr_ftest'
    
    res = grangercausalitytests(dataPair, maxlag=maxlag,verbose=verbose)
    
    bestLag = 0
    fStat = 0
    
    for lag in res.keys():
        
        if res[lag][0][test][0] > fStat:
            
            p_value = res[lag][0][test][1]
            fStat = res[lag][0][test][0]
            bestLag = lag
    print("---------------------------------------------------------------------")    
    print("We have achieved a F-Statistic of {:.2f} and p-value of {:.5f} with a Lag-order of {} for Granger Causality".format(fStat,p_value,bestLag))
    print("---------------------------------------------------------------------")
    return bestLag,fStat
        


# In[20]:


def find_best_pair_causality(allSeries):
    """" The function helps us simplify our work by first finding
    the most co-integrated pair and then checking for granger causality
    between the two series.
    """
    
    bestPair, bestTest = find_most_cointegrated_pair(allSeries,verbose=False)
    
    series1 = [ _ for _ in bestPair.keys()][0]
    series2 = [ _ for _ in bestPair.values()][0]
    
    print("-----------------------------------------------------------------")
    print("{} is most cointegrated with {} for the given timeframe".format(series1,series2))
    print("-----------------------------------------------------------------")
    
    dataPair = allSeries[[series1,series2]]
    
    bestLag,fStat = granger_causality(dataPair,verbose=False)
    
    return dataPair, bestLag


# In[21]:


def regress(Y,X,verbose=True):
    
    OLS_results = sm.RLM(Y, X, hasconst=True).fit()
    
    if verbose:
        print(OLS_results.summary())
    
    residuals = OLS_results.resid
    
    residualStatistics = [residuals.mean(), residuals.std()]
    standardResiduals = (residuals - residuals.mean()) / residuals.std()
    confidenceInterval = OLS_results.conf_int()
    
    return residuals, OLS_results.params, residualStatistics, confidenceInterval


# In[22]:


def get_residuals(allSeries):
    """" This function first finds us the pair that works the best
    for prediction from amongst our data.
    We then regress the causing series onto the prediction series
    to get the set of residuals needed.
    """
    
    dataPair, bestLag = find_best_pair_causality(allSeries)
    X = dataPair.iloc[:,1].values.astype(np.float64)[:-bestLag]
    X = stats.add_constant(X)
    Y = dataPair.iloc[:,0].shift(-bestLag).dropna().values.astype(np.float64)
   
    residuals, params, residualStatistics, confidenceInterval = regress(Y,X)
    
    results = {'Residuals':residuals, 'regressParams':params,
               'Residual Statistics':residualStatistics, 'Best Pair':dataPair.columns,
               'Best Lag': bestLag, 'Confidence Interval':confidenceInterval}

    return results


# In[23]:


results = get_residuals(standardDifferencedCrypto)


# In[24]:


residuals = results['Residuals']
regressParams = results['regressParams']
bestPair = results['Best Pair']
bestLag = results['Best Lag']
residualStatistics = results['Residual Statistics']
confidenceInterval = results['Confidence Interval']

# In[25]:

def remove_noise_by_fft(series, threshold=0.1):
    
    frequencies = fftpack.fft(series)
    cleanFreq = []
    for val in range(len(frequencies)):

        if val <= len(frequencies)*threshold:
            cleanFreq.append(frequencies[val])
        else:
            cleanFreq.append(0)

    cleanSeries = fftpack.ifft(cleanFreq)
    
    return cleanSeries

residuals = remove_noise_by_fft(residuals)

# In[25]:


plot_graphics(pd.DataFrame(residuals, columns=['Residuals']))


# In[26]:


import tensorflow.keras.backend as K

def loss(regressParams, X, Y):
    def error(y_true, y_pred):
        return custom_loss_function(y_true, y_pred, regressParams, X, Y)
    return error

def custom_loss_function(y_true, y_pred, regressParams, X, Y):
    
    """" regression constant is a list of [constant,slope]. """
    
    regressParams = tf.constant(regressParams.astype(np.float32))
    X = tf.constant(X.astype(np.float32))
    Y = tf.constant(Y.astype(np.float32))
    
    Y_pred = regressParams[0] + regressParams[1]*X + y_pred
    loss = K.mean(K.abs((Y-Y_pred)))

    return loss  


# In[27]:


def convert_to_window(timeSeries, windowSize, forecastHorizon):
    x=[]
    y=[]
    for i in range(len(timeSeries)-windowSize-forecastHorizon):
    
            x.append(timeSeries[i:i+windowSize])
            y.append(timeSeries[i+windowSize+forecastHorizon])
        
    return np.array(x),np.array(y)


# In[28]:


forecastHorizon = 3
windowSize = 20
X, Y = convert_to_window(residuals, windowSize=windowSize,
                         forecastHorizon=forecastHorizon)
X = X.reshape(X.shape[0], X.shape[1], 1)


# In[29]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional,LSTM,Dense,Dropout,TimeDistributed,Flatten


# In[30]:


def create_simple_model(regressParams, X, Y):
    """ X is the independent variable here, the stationary, differenced, and standard scaled crypto data
    Y here is the Dependent variable that we are regressing X onto 
    """
    
    WINDOW_SIZE = 20
    DROP = 0.3
    NUM_FEATURES = 1
    BATCH = 32
    
    model = Sequential(
    [  
        
    LSTM(20,return_sequences=True,recurrent_dropout=0.5,
            input_shape=(WINDOW_SIZE,NUM_FEATURES)),
    Dropout(DROP),
    Flatten(),
    Dense(16,activation='relu'),
    Dense(1)

    ])

    OPTIMIZER = tf.keras.optimizers.SGD(lr=0.1, momentum=0.8,
                                         decay=0.05, nesterov=True)
    LOSS = loss(regressParams=regressParams, X=X, Y=Y)
    model.compile(loss=LOSS, optimizer=OPTIMIZER,
                 metrics=['mean_absolute_percentage_error'])
    model.summary()
    
    return model


# In[32]:


lstmInputData = standardDifferencedCrypto.iloc[:-bestLag - forecastHorizon - windowSize,:].copy()

trainSize = int(len(lstmInputData)*0.8)
trainSetX = X[:trainSize]
trainSetY = Y[:trainSize]
testSetX = X[trainSize:]
testSetY = Y[trainSize:]

lstmTrainData = lstmInputData[:trainSize]
lstmTestData = lstmInputData[trainSize:]

model = create_simple_model(regressParams, X = lstmTrainData[bestPair[1]].values,
                                            Y = lstmTrainData[bestPair[0]].values)


# In[49]:


history = model.fit(x=X, y=Y, epochs=100, batch_size=32, verbose=False)


# In[50]:


plt.title('Loss', size=18)
plt.plot(history.history['loss']);
plt.xlabel('Epochs', size=15)
plt.ylabel('Regression Episilon', size=15)
plt.show()


# In[51]:


residualPredictions = model.predict(testSetX)*100 + 5


# In[52]:


plt.figure(figsize=(16,5))
plt.title('Test Residuals vs Predicted Residuals', size=18)
plt.plot(residualPredictions)
plt.plot(testSetY)
plt.show()


# In[53]:


plot_graphics(pd.DataFrame(residualPredictions, columns=['Residual Prediction']))


# In[54]:


predictedMean = np.add( regressParams[0],
                        regressParams[1]*lstmTestData[bestPair[1]].values,
                        residualPredictions.reshape(-1,))*20
upperInterval = np.add( confidenceInterval[0,1],
                        confidenceInterval[1,1]*lstmTestData[bestPair[1]].values,
                        residualPredictions.reshape(-1,))*20
lowerInterval = np.add( confidenceInterval[0,0],
                        confidenceInterval[1,0]*lstmTestData[bestPair[1]].values,
                        residualPredictions.reshape(-1,))*20


# In[55]:


plt.figure(figsize=(16,5))
plt.title('Test Standard Differenced Values vs Predicted Standard Differenced Values', size=15)
plt.plot(predictedMean, color='blue');
plt.plot(lstmTestData[bestPair[0]].values, color='red');
plt.fill_between(x=range(len(predictedMean)),
                 y2=upperInterval, y1=lowerInterval,
                 alpha=0.15, color='red')
plt.show()


# In[56]:


def inverse_transform(series):
    series = np.add(differencedCrypto[bestPair[0]].std()*series,
                    differencedCrypto[bestPair[0]].mean()) + testSet.shift(1)
    return series


# In[57]:


testSet = crypto[bestPair[0]].iloc[:-bestLag - forecastHorizon - windowSize][trainSize:]
predictedCryptoMean = inverse_transform(predictedMean)
upperInterval = inverse_transform(upperInterval)
lowerInterval = inverse_transform(lowerInterval)


# In[58]:



predictedArray = pd.DataFrame(
    data={

        "Test_Data": testSet,
        "Prediction": predictedCryptoMean,
        "Confidence_Interval_1":  upperInterval,
        "Confidence_Interval_2": lowerInterval,
        
    }, index=lstmTestData.index
)


# In[59]:


def plot_forecast(predictedArray, algo):

    fig, axes = plt.subplots(dpi=180, figsize=(16,5))
    axes.fill_between(
        x=predictedArray.index,
        y1=predictedArray.Confidence_Interval_1,
        y2=predictedArray.Confidence_Interval_2,
        alpha=0.15,
        color="red",
    )
    axes.plot(predictedArray.Prediction, color="red")
    axes.plot(predictedArray.Test_Data, color="blue")
    axes.set_title("Forecast of Bitcoin by {}".format(algo), size=15)
    axes.legend((predictedArray, testSet), ("Forecast", "True values"))

    plt.show()


# In[60]:


plot_forecast(predictedArray, "OLS-LSTM")

MAPE = np.mean((
    abs(predictedArray.Prediction.values - predictedArray.Test_Data.values) / predictedArray.Test_Data.values)[1:])
naiveForecast = predictedArray.Test_Data.shift(1).bfill()
naiveMAPE = np.mean((
    abs(naiveForecast.values - predictedArray.Test_Data.values) / predictedArray.Test_Data.values)[1:]
    )
rMAPE = MAPE / naiveMAPE

print("\n------------- CONGRATULATIONS !! -------------\n")
print("The OLS-LSTM model has achieved an rMAPE of: {}".format(round(rMAPE, 4)))

