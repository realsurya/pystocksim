#---------------------------------------#
#               :PROJECT:               #
#    ~Monte-Carlo Sim. for Equities~    #
#              ~Pequities~              #
#           ___           ___           #
#          /\__\         /\  \          #
#         /:/ _/_       |::\  \         #
#        /:/ /\  \      |:|:\  \        #
#       /:/ /::\  \   __|:|\:\  \       #
#      /:/_/:/\:\__\ /::::|_\:\__\      #
#      \:\/:/ /:/  / \:\~~\  \/__/      #
#       \::/ /:/  /   \:\  \            #
#        \/_/:/  /     \:\  \           #
#          /:/  /       \:\__\          #
#          \/__/         \/__/          #
#                                       #
#           Surya Manikhandan           #
#             Feb. 25, 2022             #
#    Aerospace Eng. Student @ Purdue    #
#                                       #
#        [E]:smanikha@purdue.edu        #
#  [In]:linkedin.com/in/aerospacesurya  #
#      [Git]: github.com/realsurya      #
#---------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def getDeltas(closePricesDF):
    """
    getReturns computes and returns the deltas between each element for a given array of closing prices.

    :param closePricesDF: Pandas DF containing closing prices for a given stock.

    :return deltas: Deltas between each element in closePricesDF (as numpy array).
    :return logDeltas: Natural log of deltas (as numpy array).
    :return mu: The computed Drift (mean of logDeltas).
    :return sigma: The computed Volatility (stdev of logDeltas).
    """ 

    deltas = np.array(1 + closePricesDF.pct_change())
    logDeltas = np.log(deltas)

    mu = logDeltas[1:].mean()
    sigma = logDeltas[1:].std()

    return deltas[1:], logDeltas[1:], mu, sigma


def plotDeltas(logDeltas, mu, sigma):
    """
    plotDeltas plots the distribution of the log returns. 

    :param closePricesDF: Pandas DF containing closing prices for a given stock.

    :return deltas: Deltas between each element in closePricesDF (as numpy array).
    :return logDeltas: Natural log of deltas (as numpy array).
    :return mu: The computed Drift (mean of logDeltas).
    :return sigma: The computed Volatility (stdev of logDeltas).
    """ 

    # Calc theoretical std. dist
    logDeltas.sort()
    xAxis = np.linspace(logDeltas.min(), logDeltas.max(), 500)
    normTheo = norm.pdf(xAxis, mu, sigma)

    plt.figure()
    plt.plot(xAxis, normTheo, 'k-')
    plt.hist(logDeltas, density=True)
    plt.title("Log Returns Vs. Best Normal Approximation")
    plt.xlabel("Log Returns");plt.ylabel("Frequency")
    plt.grid()
    plt.legend({"Actual Distr.","Best Approx."})
    plt.show()



