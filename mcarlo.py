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

def runMonteCarlo(startPrice, mu, sigma, propogateFor):
    """
    :runMonteCarlo: uses the Monte-Carlo Method with Geometric Brownian Motion to 
    predict future closing price of a stock in the short-term.

    Note first day returned will always be equal to startPrice (for day 0).
    Therefore, number of elements returned will be propogateFor + 1

    :param startPrice: The initial condition for this simulation (Initial price).
    :param mu: The computed Drift (mean of logDeltas).
    :param sigma: The computed Volatility (stdev of logDeltas).
    :param propogateFor: Number of time units to run the simulations for (determines size of return).

    :return futureAbs: The future absolute prices of the stock for this simulation (as numpy array).
    :return futureRel: The future changes in price of the stock for this simulation relative to start price (as numpy array).
    :return pChange: The percent change in stock price for this simulation relative to start price (as numpy array). 
    """ 
    iterations = np.arange(0,(propogateFor + 1),1)
    futureAbs = np.zeros(propogateFor + 1)

    for iteration in iterations:
        bMotion = np.random.randn()

        if iteration == 0:
            newPrice = startPrice
        else:
            newPrice = futureAbs[iteration-1] * np.exp((mu - (0.5*(sigma**2))) + (sigma * bMotion))

        futureAbs[iteration] = newPrice

    futureRel = futureAbs - startPrice
    pChange =  (futureRel/startPrice) * 100

    return futureAbs, futureRel, pChange


