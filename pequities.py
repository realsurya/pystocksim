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
import matplotlib.pyplot as plt

import returns as ret
import mcarlo as mc

def evaluate(openPricesDF, closePricesDF, propogateFor, numTrials):
    """
    evaluate will run the Monte-Carlo simulation for a certain number of trials.

    Note first day returned will always be equal to startPrice (for day 0).
    Therefore, number of elements returned will be propogateFor + 1

    :param openPricesDF: Pandas DF containing opening prices for a given stock.
    :param closePricesDF: Pandas DF containing closing prices for a given stock.
    :param propogateFor: Number of time units to run the simulations for (determines size of return).
    :param numTrials: Number of simulations to run

    :return allFuturesAbs: The final absolute prices of the stock across all simulations (as numpy array).
    :return allFuturesRel: The final price delta from starting price across all simulations (as numpy array).
    :return allFutureROI: The final percent change in stock price across all simulaitons (as numpy array). 
    """


    deltas, logDeltas, mu, sigma = ret.getDeltas(closePricesDF)
    ret.plotDeltas(logDeltas, mu, sigma)

    startPrice = closePricesDF[closePricesDF.size -1]

    allFutureAbs = np.zeros(numTrials + 1)
    allFutureRel = np.zeros(numTrials + 1)
    allFutureROI = np.zeros(numTrials + 1)

    for trial in range(1, numTrials + 1):

        futureAbs, futureRel, pChange = mc.runMonteCarlo(startPrice, mu, sigma, propogateFor)

        allFuturesAbs[trial] = futureAbs[-1]
        allFuturesRel[trial] = futureRel[-1]
        allFutureROI[trial] = pChange[-1]

    return allFuturesAbs, allFuturesRel, allFutureROI