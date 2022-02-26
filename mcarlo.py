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
    runMonteCarlo uses the Monte-Carlo Method with Geometric Brownian Motion to 
    predict future closing price of a stock in the short-term.

    :param startPrice: The initial condition for this simulation (Initial price).
    :param mu: The computed Drift (mean of logDeltas).
    :param sigma: The computed Volatility (stdev of logDeltas).
    :param propogateFor: Number of time units to run the simulations for (determines size of return).

    :return future: The future prices of the stock for this simulation (as numpy array). 
    """ 
    days = np.arange(0,propogateFor,1)
    future = np.zeros(propogateFor)

    for day in days:
        bMotion = np.random.randn()

        if day == 0:
            newPrice = startPrice
        else:
            newPrice = future[day-1] * np.exp((mu - (0.5*(sigma**2))) + (sigma * bMotion))

        future[day] = newPrice

    return future


