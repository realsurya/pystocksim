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

import returns as ret
import mcarlo as mc

def evaluate(openPricesDF, closePricesDF, propogateFor, numTrials):
    """
    :evaluate: will run the Monte-Carlo simulation for a certain number of trials.
               Strongly recommend use of jupyter-lab or notebook to view results.

    :param openPricesDF: Pandas DF containing opening prices for a given stock.
    :param closePricesDF: Pandas DF containing closing prices for a given stock.
    :param propogateFor: Number of time units to run the simulations for (determines size of return).
    :param numTrials: Number of simulations to run

    :return allFutureAbs: The final absolute prices of the stock across all simulations (as numpy array).
    :return allFutureRel: The final price delta from starting price across all simulations (as numpy array).
    :return allFutureROI: The final percent change in stock price across all simulaitons (as numpy array). 
    :return logDeltas: Natural log of deltas (as numpy array).
    :return mu: The computed Drift (mean of logDeltas).
    :return sigma: The computed Volatility (stdev of logDeltas).
    """


    deltas, logDeltas, mu, sigma = ret.getDeltas(closePricesDF)

    startPrice = closePricesDF[closePricesDF.size -1]

    allFutureAbs = np.zeros(numTrials + 1)
    allFutureRel = np.zeros(numTrials + 1)
    allFutureROI = np.zeros(numTrials + 1)

    for trial in range(1, numTrials + 1):

        futureAbs, futureRel, pChange = mc.runMonteCarlo(startPrice, mu, sigma, propogateFor)

        allFutureAbs[trial] = futureAbs[-1]
        allFutureRel[trial] = futureRel[-1]
        allFutureROI[trial] = pChange[-1]

    return allFutureAbs, allFutureRel, allFutureROI, logDeltas, mu, sigma

def insigits(allFutureAbs, allFutureRel, allFutureROI, numTrials, propogateFor):
    """
    :insigits: will provide a high-level interactive summary of the simulations.
               Strongly recommend use of jupyter-lab or notebook to view results.

    :param openPricesDF: Pandas DF containing opening prices for a given stock.
    :param closePricesDF: Pandas DF containing closing prices for a given stock.
    :param propogateFor: Number of time units to run the simulations for (determines size of return).
    :param numTrials: Number of simulations to run

    :return: N/A
    """

    plt.figure()
    plt.hist(allFutureROI, bins=100)
    plt.axvline(0, color='black')

    plt.grid()
    plt.xlabel("Percent change over period.")
    plt.ylabel("Frequency")
    plt.title("Percentage Change Over Period For All Trials")
    plt.show()

    print("\nLooking at the " + str(numTrials) + " trials conducted, here are the insights:\n")

    print("   -> Profitable Trials (ROI>0): " + str(sum(allFutureROI > 0)) + " (" + str(np.round(((sum(allFutureROI > 0)/numTrials)*100), 2)) + "%).")
    print("   -> Unprofitable Trials (ROI<0): " + str(sum(allFutureROI < 0)) + " (" + str(np.round(((sum(allFutureROI < 0)/numTrials)*100), 2)) + "%).")
    print("\n   -> Mean End Price: " + str(np.round(allFutureAbs.mean(), 4)) + " $.")
    print("   -> Mean Change: " + str(np.round(allFutureRel.mean(), 4)) + " $ (" + str(np.round(allFutureROI.mean(), 4)) + "%).")

def stats(allFutureAbs, allFutureRel, allFutureROI, numTrials, propogateFor):
    """
    :stats: Same as :insights: but not interactive

    :param openPricesDF: Pandas DF containing opening prices for a given stock.
    :param closePricesDF: Pandas DF containing closing prices for a given stock.
    :param propogateFor: Number of time units to run the simulations for (determines size of return).
    :param numTrials: Number of simulations to run

    :return numProfit: Number of trials resulting in profit (ROI > 0).
    :return pctProfit: Percent of trials resulting in a Profit.
    :return numloss: Number of trials resulting in profit (ROI > 0).
    :return pctlos: Percent of trials resulting in a Profit.
    :return meanEnd: Mean ending price of stock.
    :return meanChg: Mean change in price of stock.
    :return meanPct: Mean percent change in price of stock.
    """
    numProfit = sum(allFutureROI > 0)
    pctProfit = ((sum(allFutureROI > 0)/numTrials)*100)

    numLoss = sum(allFutureROI < 0)
    pctLoss = ((sum(allFutureROI < 0)/numTrials)*100)

    meanEnd = allFutureAbs.mean()
    meanChg = allFutureRel.mean()
    meanPct = allFutureROI.mean()

    return numProfit, pctProfit, numLoss, pctLoss, meanEnd, meanChg, meanPct