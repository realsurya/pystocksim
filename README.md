# Pystocksim

PystockSim is a python program for simulating and calculating probability of profit for stock trading. PystockSim can execute thousands of Monte-Carlo simulations with Geometric Brownian Motion to predict future closing price of a stock in the short-term.

# Assumptions & Limitations

- Drift and volatility of stock are constant through timeframe. Therefore, this model is not viable for long-term predictions. Recommend limiting propogation to 14 days.

- The natural log of historical returns are assumed follow an approximately normal distribution (necessary for GBM).

- Geometric Brownian Motion (GBM) can be used to simulate stock price movement.

- Fees and/or commissions imposed by brokerages are not considered.

- Earnings, news, and such fuzzy factors not considered.

- Dividends are not considered.

- Stock splits are not considered.

*This program has not been verified by any financial experts / institutions. The results of this program are not to be used as financial advice, just as an additional tool to better make informed purchasing decisions. As you may see from the assumptions, one must take the results with a grain of salt.*

# Theoretical Basis

Much of the functionality of this program is based on these excellent studies below:

> Abidin, S. N., & Jaffar, M. M. (2014). Forecasting share prices of small size companies in Bursa Malaysia using geometric brownian motion. *Applied Mathematics & Information Sciences*, *8*(1), 107–112. https://doi.org/10.12785/amis/080112

> Liden, J. (2018). Stock Price Predictions using a Geometric Brownian Motion. *Uppsala University*.

# Quick Start

```python
import yfinance as yf
import numpy as np

import pystocksim
import returns as ret
```

###### Get Stock Historical Data

```python
# For this example, I'm using yfinance to get open and close prices for Tesla stock.
stock = yf.Ticker("TSLA")
```

```python
historicals = stock.history(period="1mo", interval="1d")
```

```python
opens = historicals.iloc[:,0]
closes = historicals.iloc[:,3]
```

###### Run Monte-Carlo Simulations.

```python
propogateFor = 14
numTrials = 50000

allFutureAbs, allFutureRel, allFutureROI, logDeltas, mu, sigma = pystocksim.evaluate(opens, closes, propogateFor, numTrials)
```

###### Gain Insights Fast.

```python
pystocksim.insigits(allFutureAbs, allFutureRel, allFutureROI, numTrials, propogateFor)![png](output_7_0.png)
```

```
Looking at the 50000 trials conducted, here are the insights:

   -> Profitable Trials (ROI>0): 13228 (26.46%).
   -> Unprofitable Trials (ROI<0): 36772 (73.54%).

   -> Mean End Price: 735.6779 $.
   -> Mean Change: -74.1759 $ (-9.159%).
```

```python
# Or use the non-interactive function to directly access these statistics:
numProfit, pctProfit, numLoss, pctLoss, meanEnd, meanChg, meanPct = pystocksim.stats(allFutureAbs, allFutureRel, allFutureROI, numTrials, propogateFor)
# Or just do your own analysis :)
```

# Further Documentation

The quick start guide can also be found in ipynb (Jupyter Lab) format in this repo. This includes the plots pystocksim will output. Additionally, documentation for each function can be found in the respective docstring.

# Future Expandability

- Potentially implement numba & jit to speed up calculation by 100x. The only limitation is the use of pandas dataframe, which numba does not support.

- Ability to screen many tickers at once.

- Adding package to pypi for pip installation.

# Issue Reporting

Please report issues through the Github repository page.
