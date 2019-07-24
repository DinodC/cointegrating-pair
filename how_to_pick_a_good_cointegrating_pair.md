
# How To Pick A Good Cointegrating Pair

## Introduction
A time series is considered stationary if its probability distribution does not change over time.
If the price series of a security is stationary, then it would be a suitable candidate for a mean-reversion trading strategy.
However, most security price series are not stationary: they seem to follow a lognormal random walk; and drift farther and farther away from the initial value. 

We need to find a pair of securities such that the combination of the two is stationary, e.g. buying a security and shorting another.
Two securities that form a stationary or cointegrating pair are often from the same industry group such as Coca-Cola Company and PepsiCo.
In this article, we illustrate how to pick a good cointegrating pair by applying the augmented Dickey-Fuller test to security pairs to check for cointegration.

## Step-by-step
We will proceed as follows:
1. Determine The Pairs: We present the security pairs to analyze.
2. Prepare The Data: We pull and process securities' open-high-low-close-volume (OHLCV) data.
3. Calculate The Spread: We apply the ordinary least squares (OLS) method to calculate the spread between two securities.
4. Check For Cointegration: We use the augmented Dickey-Fuller test to check if two securities form a stationary or cointegrating pair.

## Determine The Pairs
Below are the pairs of securities which we will check for cointegration:

### 1. Gold
Gold-themed exchange traded funds (ETF):
- VanEck Vectors Gold Miners ETF (GDX): ETF which tracks a basket of gold-mining companies.
- SPDR Gold Shares (GLD): ETF which replicates the price of gold bullion.

### 2. Fast Food
Companies serving fast food:
- McDonald's Corporation (MCD): Fast food company which gave the whole world classics like *Big Mac*, *Hot Fudge Sundae*, and *Happy Meal*.
- YUM! Brands, Inc. (YUM): Fast food company which operates Taco Bell, KFC and Pizza Hut.
    
### 3. Cryptocurrencies
Digital currencies:
- Bitcoin USD (BTC-USD): A decentralized cryptocurrency that can be sent from user to user on the peer-to-peer bitcoin network established in 2009.
- Ethereum USD (ETH-USD): An open source, public, blockchain-based distributed computing platform and operating system released in 2015.

## Prepare The Data
In this section, we illustrate download and preparation of securities' price series.
We pull the securities' historical OHLCV data from [Yahoo Finance](https://sg.finance.yahoo.com).
We select the adjusted close prices for each security and create a new Dataframe object.

Import packages


```python
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

Magic


```python
%matplotlib inline
```

Set tickers list


```python
tickers = ['GDX', 'GLD', 'MCD', 'YUM', 'BTC-USD', 'ETH-USD']
```

Pull OHLCV data


```python
# Initialize list of DataFrames
df_list = []

# Load DataFrames
for i in tickers:
    
    # Load data
    df = pd.read_csv(i + '.csv', index_col=0, parse_dates=True)    
    
    # Set multi-level columns
    df.columns = pd.MultiIndex.from_product([[i], ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    
    # Update list
    df_list.append(df)
    
# Merge DataFrames
data = pd.concat(df_list, axis=1, join='inner')

# Drop NaNs
data.dropna(inplace=True)
```

Inspect OHLCV data


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">GDX</th>
      <th colspan="4" halign="left">GLD</th>
      <th>...</th>
      <th colspan="4" halign="left">BTC-USD</th>
      <th colspan="6" halign="left">ETH-USD</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>...</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-08-06</th>
      <td>13.21</td>
      <td>13.69</td>
      <td>13.11</td>
      <td>13.36</td>
      <td>13.033523</td>
      <td>69121200</td>
      <td>104.150002</td>
      <td>104.860001</td>
      <td>104.139999</td>
      <td>104.389999</td>
      <td>...</td>
      <td>274.279999</td>
      <td>277.890015</td>
      <td>277.890015</td>
      <td>11919665</td>
      <td>0.6747</td>
      <td>3.000</td>
      <td>0.6747</td>
      <td>3.000</td>
      <td>3.000</td>
      <td>371</td>
    </tr>
    <tr>
      <th>2015-08-07</th>
      <td>13.42</td>
      <td>13.85</td>
      <td>13.33</td>
      <td>13.40</td>
      <td>13.072546</td>
      <td>50618200</td>
      <td>104.559998</td>
      <td>105.379997</td>
      <td>104.550003</td>
      <td>104.650002</td>
      <td>...</td>
      <td>257.420013</td>
      <td>258.600006</td>
      <td>258.600006</td>
      <td>22308123</td>
      <td>3.0000</td>
      <td>3.000</td>
      <td>0.1500</td>
      <td>1.200</td>
      <td>1.200</td>
      <td>1438</td>
    </tr>
    <tr>
      <th>2015-08-10</th>
      <td>13.57</td>
      <td>14.29</td>
      <td>13.36</td>
      <td>14.27</td>
      <td>13.921287</td>
      <td>91376800</td>
      <td>105.029999</td>
      <td>106.269997</td>
      <td>104.919998</td>
      <td>105.720001</td>
      <td>...</td>
      <td>261.440002</td>
      <td>269.029999</td>
      <td>269.029999</td>
      <td>13681939</td>
      <td>1.2000</td>
      <td>1.200</td>
      <td>0.6504</td>
      <td>0.990</td>
      <td>0.990</td>
      <td>7419</td>
    </tr>
    <tr>
      <th>2015-08-11</th>
      <td>14.44</td>
      <td>14.53</td>
      <td>13.94</td>
      <td>14.53</td>
      <td>14.174931</td>
      <td>53731900</td>
      <td>106.489998</td>
      <td>106.629997</td>
      <td>105.769997</td>
      <td>106.260002</td>
      <td>...</td>
      <td>263.660004</td>
      <td>267.660004</td>
      <td>267.660004</td>
      <td>15232934</td>
      <td>0.9900</td>
      <td>1.288</td>
      <td>0.9050</td>
      <td>1.288</td>
      <td>1.288</td>
      <td>2376</td>
    </tr>
    <tr>
      <th>2015-08-12</th>
      <td>14.81</td>
      <td>15.53</td>
      <td>14.78</td>
      <td>15.52</td>
      <td>15.140740</td>
      <td>123217200</td>
      <td>106.989998</td>
      <td>107.910004</td>
      <td>106.930000</td>
      <td>107.750000</td>
      <td>...</td>
      <td>261.279999</td>
      <td>263.440002</td>
      <td>263.440002</td>
      <td>14962211</td>
      <td>1.2880</td>
      <td>1.885</td>
      <td>1.2630</td>
      <td>1.885</td>
      <td>1.885</td>
      <td>4923</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>



For WordPress


```python
data.head()[['GDX', 'GLD']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">GDX</th>
      <th colspan="6" halign="left">GLD</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-08-06</th>
      <td>13.21</td>
      <td>13.69</td>
      <td>13.11</td>
      <td>13.36</td>
      <td>13.033523</td>
      <td>69121200</td>
      <td>104.150002</td>
      <td>104.860001</td>
      <td>104.139999</td>
      <td>104.389999</td>
      <td>104.389999</td>
      <td>3908100</td>
    </tr>
    <tr>
      <th>2015-08-07</th>
      <td>13.42</td>
      <td>13.85</td>
      <td>13.33</td>
      <td>13.40</td>
      <td>13.072546</td>
      <td>50618200</td>
      <td>104.559998</td>
      <td>105.379997</td>
      <td>104.550003</td>
      <td>104.650002</td>
      <td>104.650002</td>
      <td>4400900</td>
    </tr>
    <tr>
      <th>2015-08-10</th>
      <td>13.57</td>
      <td>14.29</td>
      <td>13.36</td>
      <td>14.27</td>
      <td>13.921287</td>
      <td>91376800</td>
      <td>105.029999</td>
      <td>106.269997</td>
      <td>104.919998</td>
      <td>105.720001</td>
      <td>105.720001</td>
      <td>5892600</td>
    </tr>
    <tr>
      <th>2015-08-11</th>
      <td>14.44</td>
      <td>14.53</td>
      <td>13.94</td>
      <td>14.53</td>
      <td>14.174931</td>
      <td>53731900</td>
      <td>106.489998</td>
      <td>106.629997</td>
      <td>105.769997</td>
      <td>106.260002</td>
      <td>106.260002</td>
      <td>4060900</td>
    </tr>
    <tr>
      <th>2015-08-12</th>
      <td>14.81</td>
      <td>15.53</td>
      <td>14.78</td>
      <td>15.52</td>
      <td>15.140740</td>
      <td>123217200</td>
      <td>106.989998</td>
      <td>107.910004</td>
      <td>106.930000</td>
      <td>107.750000</td>
      <td>107.750000</td>
      <td>10022500</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">GDX</th>
      <th colspan="4" halign="left">GLD</th>
      <th>...</th>
      <th colspan="4" halign="left">BTC-USD</th>
      <th colspan="6" halign="left">ETH-USD</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>...</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-08</th>
      <td>25.450001</td>
      <td>25.610001</td>
      <td>25.209999</td>
      <td>25.420000</td>
      <td>25.420000</td>
      <td>40606100</td>
      <td>132.179993</td>
      <td>132.339996</td>
      <td>131.279999</td>
      <td>131.289993</td>
      <td>...</td>
      <td>12117.309570</td>
      <td>12567.019531</td>
      <td>12567.019531</td>
      <td>993891866</td>
      <td>313.339996</td>
      <td>318.320007</td>
      <td>303.089996</td>
      <td>307.890015</td>
      <td>307.890015</td>
      <td>125850428</td>
    </tr>
    <tr>
      <th>2019-07-09</th>
      <td>25.330000</td>
      <td>25.660000</td>
      <td>25.209999</td>
      <td>25.650000</td>
      <td>25.650000</td>
      <td>37529700</td>
      <td>131.429993</td>
      <td>132.100006</td>
      <td>131.160004</td>
      <td>131.750000</td>
      <td>...</td>
      <td>11569.940430</td>
      <td>12099.120117</td>
      <td>12099.120117</td>
      <td>1554955347</td>
      <td>307.890015</td>
      <td>314.739990</td>
      <td>281.619995</td>
      <td>288.640015</td>
      <td>288.640015</td>
      <td>180940011</td>
    </tr>
    <tr>
      <th>2019-07-10</th>
      <td>26.020000</td>
      <td>26.230000</td>
      <td>25.770000</td>
      <td>26.200001</td>
      <td>26.200001</td>
      <td>56454300</td>
      <td>132.940002</td>
      <td>133.869995</td>
      <td>132.350006</td>
      <td>133.830002</td>
      <td>...</td>
      <td>11002.389648</td>
      <td>11343.120117</td>
      <td>11343.120117</td>
      <td>1185222449</td>
      <td>288.640015</td>
      <td>288.660004</td>
      <td>263.000000</td>
      <td>268.559998</td>
      <td>268.559998</td>
      <td>171079615</td>
    </tr>
    <tr>
      <th>2019-07-11</th>
      <td>26.129999</td>
      <td>26.280001</td>
      <td>25.719999</td>
      <td>25.940001</td>
      <td>25.940001</td>
      <td>54013400</td>
      <td>133.580002</td>
      <td>133.699997</td>
      <td>132.410004</td>
      <td>132.699997</td>
      <td>...</td>
      <td>11096.610352</td>
      <td>11797.370117</td>
      <td>11797.370117</td>
      <td>647690095</td>
      <td>268.559998</td>
      <td>279.059998</td>
      <td>266.459991</td>
      <td>275.410004</td>
      <td>275.410004</td>
      <td>76685542</td>
    </tr>
    <tr>
      <th>2019-07-12</th>
      <td>26.000000</td>
      <td>26.250000</td>
      <td>25.870001</td>
      <td>26.209999</td>
      <td>26.209999</td>
      <td>31795200</td>
      <td>132.889999</td>
      <td>133.690002</td>
      <td>132.529999</td>
      <td>133.529999</td>
      <td>...</td>
      <td>10827.530273</td>
      <td>11363.969727</td>
      <td>11363.969727</td>
      <td>668325183</td>
      <td>275.410004</td>
      <td>275.720001</td>
      <td>261.809998</td>
      <td>268.940002</td>
      <td>268.940002</td>
      <td>66861426</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>



For WordPress


```python
data.tail()[['GDX', 'GLD']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">GDX</th>
      <th colspan="6" halign="left">GLD</th>
    </tr>
    <tr>
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-08</th>
      <td>25.450001</td>
      <td>25.610001</td>
      <td>25.209999</td>
      <td>25.420000</td>
      <td>25.420000</td>
      <td>40606100</td>
      <td>132.179993</td>
      <td>132.339996</td>
      <td>131.279999</td>
      <td>131.289993</td>
      <td>131.289993</td>
      <td>8028900</td>
    </tr>
    <tr>
      <th>2019-07-09</th>
      <td>25.330000</td>
      <td>25.660000</td>
      <td>25.209999</td>
      <td>25.650000</td>
      <td>25.650000</td>
      <td>37529700</td>
      <td>131.429993</td>
      <td>132.100006</td>
      <td>131.160004</td>
      <td>131.750000</td>
      <td>131.750000</td>
      <td>6633400</td>
    </tr>
    <tr>
      <th>2019-07-10</th>
      <td>26.020000</td>
      <td>26.230000</td>
      <td>25.770000</td>
      <td>26.200001</td>
      <td>26.200001</td>
      <td>56454300</td>
      <td>132.940002</td>
      <td>133.869995</td>
      <td>132.350006</td>
      <td>133.830002</td>
      <td>133.830002</td>
      <td>13920600</td>
    </tr>
    <tr>
      <th>2019-07-11</th>
      <td>26.129999</td>
      <td>26.280001</td>
      <td>25.719999</td>
      <td>25.940001</td>
      <td>25.940001</td>
      <td>54013400</td>
      <td>133.580002</td>
      <td>133.699997</td>
      <td>132.410004</td>
      <td>132.699997</td>
      <td>132.699997</td>
      <td>7535100</td>
    </tr>
    <tr>
      <th>2019-07-12</th>
      <td>26.000000</td>
      <td>26.250000</td>
      <td>25.870001</td>
      <td>26.209999</td>
      <td>26.209999</td>
      <td>31795200</td>
      <td>132.889999</td>
      <td>133.690002</td>
      <td>132.529999</td>
      <td>133.529999</td>
      <td>133.529999</td>
      <td>6308600</td>
    </tr>
  </tbody>
</table>
</div>



Select adjusted close prices


```python
# Initialize dictionary of adjusted close
close_dict = {}

# Update dictionary
for i in tickers:
    close_dict[i] = data[i]['Adj Close']
    
# Create DataFrame
close = pd.DataFrame(close_dict)
```

Inspect adjusted close prices


```python
close.head()
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
      <th>GDX</th>
      <th>GLD</th>
      <th>MCD</th>
      <th>YUM</th>
      <th>BTC-USD</th>
      <th>ETH-USD</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-08-06</th>
      <td>13.033523</td>
      <td>104.389999</td>
      <td>89.038742</td>
      <td>57.964733</td>
      <td>277.890015</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>2015-08-07</th>
      <td>13.072546</td>
      <td>104.650002</td>
      <td>88.653343</td>
      <td>57.859062</td>
      <td>258.600006</td>
      <td>1.200</td>
    </tr>
    <tr>
      <th>2015-08-10</th>
      <td>13.921287</td>
      <td>105.720001</td>
      <td>89.074577</td>
      <td>57.997757</td>
      <td>269.029999</td>
      <td>0.990</td>
    </tr>
    <tr>
      <th>2015-08-11</th>
      <td>14.174931</td>
      <td>106.260002</td>
      <td>88.554787</td>
      <td>55.171165</td>
      <td>267.660004</td>
      <td>1.288</td>
    </tr>
    <tr>
      <th>2015-08-12</th>
      <td>15.140740</td>
      <td>107.750000</td>
      <td>88.079796</td>
      <td>53.282372</td>
      <td>263.440002</td>
      <td>1.885</td>
    </tr>
  </tbody>
</table>
</div>




```python
close.tail()
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
      <th>GDX</th>
      <th>GLD</th>
      <th>MCD</th>
      <th>YUM</th>
      <th>BTC-USD</th>
      <th>ETH-USD</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-08</th>
      <td>25.420000</td>
      <td>131.289993</td>
      <td>212.160004</td>
      <td>110.050003</td>
      <td>12567.019531</td>
      <td>307.890015</td>
    </tr>
    <tr>
      <th>2019-07-09</th>
      <td>25.650000</td>
      <td>131.750000</td>
      <td>212.089996</td>
      <td>110.489998</td>
      <td>12099.120117</td>
      <td>288.640015</td>
    </tr>
    <tr>
      <th>2019-07-10</th>
      <td>26.200001</td>
      <td>133.830002</td>
      <td>213.000000</td>
      <td>110.980003</td>
      <td>11343.120117</td>
      <td>268.559998</td>
    </tr>
    <tr>
      <th>2019-07-11</th>
      <td>25.940001</td>
      <td>132.699997</td>
      <td>212.690002</td>
      <td>111.500000</td>
      <td>11797.370117</td>
      <td>275.410004</td>
    </tr>
    <tr>
      <th>2019-07-12</th>
      <td>26.209999</td>
      <td>133.529999</td>
      <td>212.990005</td>
      <td>111.050003</td>
      <td>11363.969727</td>
      <td>268.940002</td>
    </tr>
  </tbody>
</table>
</div>



Consider the training set from 2018 to present


```python
training = close['2018-01-01':'2020-01-01'].copy()
```

Inspect training set


```python
training.head()
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
      <th>GDX</th>
      <th>GLD</th>
      <th>MCD</th>
      <th>YUM</th>
      <th>BTC-USD</th>
      <th>ETH-USD</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>23.694632</td>
      <td>125.150002</td>
      <td>166.895370</td>
      <td>79.503891</td>
      <td>14754.129883</td>
      <td>861.969971</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>23.445948</td>
      <td>124.820000</td>
      <td>166.192001</td>
      <td>79.435699</td>
      <td>15156.620117</td>
      <td>941.099976</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>23.595158</td>
      <td>125.459999</td>
      <td>167.357834</td>
      <td>80.244370</td>
      <td>15180.080078</td>
      <td>944.830017</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>23.545422</td>
      <td>125.330002</td>
      <td>167.695084</td>
      <td>80.712036</td>
      <td>16954.779297</td>
      <td>967.130005</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>23.296738</td>
      <td>125.309998</td>
      <td>167.579422</td>
      <td>80.848442</td>
      <td>14976.169922</td>
      <td>1136.109985</td>
    </tr>
  </tbody>
</table>
</div>




```python
training.tail()
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
      <th>GDX</th>
      <th>GLD</th>
      <th>MCD</th>
      <th>YUM</th>
      <th>BTC-USD</th>
      <th>ETH-USD</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-08</th>
      <td>25.420000</td>
      <td>131.289993</td>
      <td>212.160004</td>
      <td>110.050003</td>
      <td>12567.019531</td>
      <td>307.890015</td>
    </tr>
    <tr>
      <th>2019-07-09</th>
      <td>25.650000</td>
      <td>131.750000</td>
      <td>212.089996</td>
      <td>110.489998</td>
      <td>12099.120117</td>
      <td>288.640015</td>
    </tr>
    <tr>
      <th>2019-07-10</th>
      <td>26.200001</td>
      <td>133.830002</td>
      <td>213.000000</td>
      <td>110.980003</td>
      <td>11343.120117</td>
      <td>268.559998</td>
    </tr>
    <tr>
      <th>2019-07-11</th>
      <td>25.940001</td>
      <td>132.699997</td>
      <td>212.690002</td>
      <td>111.500000</td>
      <td>11797.370117</td>
      <td>275.410004</td>
    </tr>
    <tr>
      <th>2019-07-12</th>
      <td>26.209999</td>
      <td>133.529999</td>
      <td>212.990005</td>
      <td>111.050003</td>
      <td>11363.969727</td>
      <td>268.940002</td>
    </tr>
  </tbody>
</table>
</div>



Calculate the number of pairs


```python
no_pairs = round(0.5 * len(tickers))
```

Plot the adjusted close prices


```python
plt.figure(figsize=(20, 20))

for i in range(no_pairs):
    # Primary axis
    color = 'tab:blue'
    ax1 = plt.subplot(3, 1, i+1)
    plt.plot(training[tickers[2*i]], color=color)
    ax1.set_ylabel('Adjusted Close Price of ' + tickers[2*i], color=color)
    ax1.tick_params(labelcolor=color)
    
    # Secondary axis 
    color = 'tab:orange'
    ax2 = ax1.twinx()
    plt.plot(training[tickers[2*i+1]], color=color)
    ax2.set_ylabel('Adjusted Close Price of ' + tickers[2*i+1], color=color)
    ax2.tick_params(labelcolor=color)
    
    # Both axis
    plt.xlim([training.index[0], training.index[-1]])
    plt.title('Adjusted Close Prices of ' + tickers[2*i] + ' and ' + tickers[2*i+1])
```


![png](output_33_0.png)


## Calculate The Spread
In this section, we calculate the spread between the securities.
We apply the OLS method between the securities to calculate for the hedge ratio.
We standardize the spread by subtracting the mean and scaling by the standard deviation of the spread.

Calculate the spread between each pair


```python
# Initialize the spread list
spread_list = []

for i in range(no_pairs):
    # Run an OLS regression between the pairs
    model = sm.regression.linear_model.OLS(training[tickers[2*i]], training[tickers[2*i+1]])

    # Calculate the hedge ratio
    results = model.fit()
    hedge_ratio = results.params[0]
    
    # Calculate the spread
    spread = training[tickers[2*i]] - hedge_ratio * training[tickers[2*i+1]]
    
    # Mean and standard deviation of the spread
    spread_mean = spread.mean()
    spread_std = spread.std()
    
    # Standardize the spread
    z_score = (spread - spread_mean) / spread_std
    
    # Update the spread list
    spread_list.append(z_score)
```

Plot the spread


```python
plt.figure(figsize=(20, 20))

for i in range(no_pairs):
    plt.subplot(3, 1, i+1)
    plt.plot(spread_list[i])
    plt.xlim([spread.index[0], spread.index[-1]])
    plt.ylim([-3, 3])
    plt.title('Spread between ' + tickers[2*i] + ' and ' + tickers[2*i+1])
```


![png](output_38_0.png)


## Check For Cointegration
In this section, we test if two securities form a stationary or cointegrating pair.
We use the augmented Dickey-Fuller (ADF) test where we have the following:
1. The **null hypothesis** is that a unit root is present in the price series, it is **non-stationary**.
2. The **alternative** is that unit root is not present in the prices series, it is **stationary**.

Run cointegration check using augmented Dickey-Fuller test


```python
# Initialize stats
stats_list = []

for i in range(len(spread_list)):
    
    # ADF test
    stats = adfuller(spread_list[i])
    
    # Update stats
    stats_list.append(stats)
```

Set the pairs


```python
# Initialize pairs
pairs = []

for i in range(no_pairs):
    # Update pairs
    pairs.append(tickers[2*i] + '/' + tickers[2*i+1])
```

Create stats DataFrame


```python
# Initialize dict
stats_dict = {}

for i in range(no_pairs):
    
    # Update dict
    stats_dict[pairs[i]] = [stats_list[i][0],
                            stats_list[i][1],
                            stats_list[i][4]['1%'], stats_list[i][4]['5%'], stats_list[i][4]['10%']]

# Create DataFrame
stats_df = pd.DataFrame(stats_dict,
                          index=['ADF Statistic', 'P-value', '1%', '5%', '10%'])
```

Inspect


```python
stats_df
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
      <th>GDX/GLD</th>
      <th>MCD/YUM</th>
      <th>BTC-USD/ETH-USD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADF Statistic</th>
      <td>-3.386075</td>
      <td>-3.072452</td>
      <td>-2.161601</td>
    </tr>
    <tr>
      <th>P-value</th>
      <td>0.011443</td>
      <td>0.028660</td>
      <td>0.220476</td>
    </tr>
    <tr>
      <th>1%</th>
      <td>-3.448344</td>
      <td>-3.447815</td>
      <td>-3.448344</td>
    </tr>
    <tr>
      <th>5%</th>
      <td>-2.869469</td>
      <td>-2.869237</td>
      <td>-2.869469</td>
    </tr>
    <tr>
      <th>10%</th>
      <td>-2.570994</td>
      <td>-2.570870</td>
      <td>-2.570994</td>
    </tr>
  </tbody>
</table>
</div>



Remarks:
1. For the spread between GDX and GLD, the ADF statistic is -3.39 which is lower than the 1% critical value -3.45, which means that there is a better than 99% probability that the **spread between GDX and GLD is stationary**. 
2. For the spread between MCD and YUM, the ADF statistic is -3.07 is between the 1% critical value -3.45 and 5% critical value of -2.87, which means that there is a better than 95% probability that the **spread between MCD and YUM is stationary**.
3. For the spread between BTC-USD and ETH-USD, the ADF statistic is -2.16 which is higher than the critical values, which means that the **spread between BTC-USD and ETH-USD is not stationary**.

## Conclusion
In this article, we demonstrated how to form a a good cointegrating pair of securities.
We used the OLS method to determine the hedge ratio between securities; and the ADF test to check for stationarity.
The results suggest the following: cointegraing pairs could be formed within gold (GDX and GLD) and fast food securities (MCD and YUM); and cointegrating pairs could not be formed within cryptocurrencies (BTC-USD and ETH-USD).
