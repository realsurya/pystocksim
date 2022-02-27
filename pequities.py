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

    deltas, logDeltas, mu, sigma = ret.getDeltas(closes)
    ret.plotDeltas(logDeltas, mu, sigma)

    startPrice = closePricesDF[closePricesDF.size -1]

    allFutureAbs, allFutureRel, allFutureROI = np.zeros(numTrials + 1)

    for trial in range(1, numTrials + 1):
        futureAbs, futureRel, pChange = mc.runMonteCarlo(startPrice, mu, sigma, propogateFor)