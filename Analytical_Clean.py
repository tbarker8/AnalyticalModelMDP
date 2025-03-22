from scipy.stats import binom
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from steadystate import getSteadyStateDist
import itertools

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# functions:
#   -findEntropy
#   -findAllocationsAB
#   -find Max
#   -

# from allocation of A/B
#   - find max possible growth
#   -

modifyP = 0
gamma = 0.95


def generatePDF(N_large, ps):
    conc1PDFs = []
    conc2PDFs = []
    conc3PDFs = []
    for i in range(N_large + 1):
        x = np.arange(0, i + 1)
        conc1PDFs.append(binom.pmf(x, i, ps[0]))
        conc2PDFs.append(binom.pmf(x, i, ps[1]))
        conc3PDFs.append(binom.pmf(x, i, ps[2]))
    return conc1PDFs, conc2PDFs, conc3PDFs


# MI = MIA + MIB
# MIA = sum x sum y pxy * log(pxy/(px*py))
# since the chance of seeing any specific A or B concentration is the same = 1/3 px = 1/3
def CaluclateMIs(conc1PDFs, conc2PDFs, conc3PDFs, n):
    def calculateMarginalY(pdf1, pdf2, pdf3):
        MarginalY = []
        for i in range(len(pdf1)):
            MarginalY.append((pdf1[i] + pdf2[i] + pdf3[i]) / 3)
        return MarginalY

    px = 1 / 3
    MIsums = []
    for i in range(n + 1):
        MIsums.append(0.0)
        currConc1PDF = conc1PDFs[i]
        currConc2PDF = conc2PDFs[i]
        currConc3PDF = conc3PDFs[i]
        MargY = calculateMarginalY(currConc1PDF, currConc2PDF, currConc3PDF)
        for j in range(len(currConc1PDF)):
            MIsums[i] += currConc1PDF[j] * px * math.log2(currConc1PDF[j] * px / (px * MargY[j]))
        for j in range(len(currConc2PDF)):
            MIsums[i] += currConc2PDF[j] * px * math.log2(currConc2PDF[j] * px / (px * MargY[j]))
        for j in range(len(currConc3PDF)):
            MIsums[i] += currConc3PDF[j] * px * math.log2(currConc3PDF[j] * px / (px * MargY[j]))
    return MIsums


# sums the rewards of the individual
def sumRewards(rewards):
    sum1 = 0.0
    for i in range(len(rewards)):
        sum1 += sum(rewards[i])
    return sum1


def findRewardFromAllocs(Allocs, nA, nB, diff, conc1PDFs, conc2PDFs, conc3PDFs):
    def calcReward(alloc, pA1, pA2, pA3, pB1, pB2, pB3):
        pASum = pA1 + pA2 + pA3
        pBSum = pB1 + pB2 + pB3

        PA1b = pA1 / pASum
        PA2b = pA2 / pASum
        PA3b = pA3 / pASum

        PB1l = pB1 / pBSum
        PB2l = pB2 / pBSum
        PB3l = pB3 / pBSum

        if diff == 0:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == 1:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (2 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (2 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (3 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (2 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (2 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (3 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (2 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (2 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (3 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == -1:
            ml = (2 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (2 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (3 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (2 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (2 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (3 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (2 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (2 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (3 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff >= 2:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (3 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (2 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (3 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (3 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (2 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (3 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (3 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (2 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (3 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff <= -2:
            ml = (3 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (2 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (3 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (3 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (2 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (3 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (3 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (2 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (3 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)
        if alloc == -1:
            reward = ml
        elif alloc == 0:
            reward = dm
        elif alloc == 1:
            reward = mr
        return alloc, reward
    # ps = [.33333, 0.5, .6]
    # A
    # x = np.arange(0, nA + 1)
    PDFA1 = conc1PDFs[nA]
    PDFA2 = conc2PDFs[nA]
    PDFA3 = conc3PDFs[nA]

    # B
    # x = np.arange(0, nB + 1)
    PDFB1 = conc1PDFs[nB]
    PDFB2 = conc2PDFs[nB]
    PDFB3 = conc3PDFs[nB]

    allocs = []
    rewards = []
    allChance = 0.0
    for i in range(len(PDFA1)):
        rewards.append([])
        for j in range(len(PDFB1)):
            alloc, reward = calcReward(Allocs[i][j], PDFA1[i], PDFA2[i], PDFA3[i], PDFB1[j], PDFB2[j], PDFB3[j])
            # allocs[i].append(alloc)
            chance = ((1 / 3) * (PDFA1[i] + PDFA2[i] + PDFA3[i])) * ((1 / 3) * (PDFB1[j] + PDFB2[j] + PDFB3[j]))
            allChance += chance
            rewards[i].append(reward * chance)

    return Allocs, sumRewards(rewards)

def findAllocationAB(nA, nB, diff, conc1PDFs, conc2PDFs, conc3PDFs):
    def calcReward(pA1, pA2, pA3, pB1, pB2, pB3):
        pASum = pA1 + pA2 + pA3
        pBSum = pB1 + pB2 + pB3

        PA1b = pA1 / pASum
        PA2b = pA2 / pASum
        PA3b = pA3 / pASum

        PB1l = pB1 / pBSum
        PB2l = pB2 / pBSum
        PB3l = pB3 / pBSum

        if diff == 0:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == 1:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (2 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (2 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (3 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (2 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (2 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (3 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (2 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (2 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (3 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff == -1:
            ml = (2 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (2 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (3 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (2 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (2 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (3 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (2 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (2 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (3 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff >= 2:
            ml = (1 * PA2b * PB2l) + (2 * PA1b * PB3l) + (3 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (1 * PA1b * PB2l) + (2 * PA3b * PB3l) \
                 + (2 * PA2b * PB3l) + (3 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (1 * PA3b * PB1l) + (2 * PA2b * PB2l) + (3 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (1 * PA2b * PB1l) + (2 * PA1b * PB2l) \
                 + (2 * PA3b * PB2l) + (3 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (1 * PA1b * PB3l) + (2 * PA3b * PB1l) + (3 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (1 * PA3b * PB3l) + (2 * PA2b * PB1l) \
                 + (2 * PA1b * PB1l) + (3 * PA3b * PB2l) + (1 * PA2b * PB3l)

        elif diff <= -2:
            ml = (3 * PA2b * PB2l) + (2 * PA1b * PB3l) + (1 * PA3b * PB1l) \
                 + (3 * PA2b * PB1l) + (2 * PA1b * PB2l) + (1 * PA3b * PB3l) \
                 + (3 * PA2b * PB3l) + (2 * PA1b * PB1l) + (1 * PA3b * PB2l)

            dm = (3 * PA3b * PB1l) + (2 * PA2b * PB2l) + (1 * PA1b * PB3l) \
                 + (3 * PA3b * PB3l) + (2 * PA2b * PB1l) + (1 * PA1b * PB2l) \
                 + (3 * PA3b * PB2l) + (2 * PA2b * PB3l) + (1 * PA1b * PB1l)

            mr = (3 * PA1b * PB3l) + (2 * PA3b * PB1l) + (1 * PA2b * PB2l) \
                 + (3 * PA1b * PB2l) + (2 * PA3b * PB3l) + (1 * PA2b * PB1l) \
                 + (3 * PA1b * PB1l) + (2 * PA3b * PB2l) + (1 * PA2b * PB3l)
        if mr > ml:
            if mr > dm:
                # move right
                return 1, mr
            elif mr < dm:
                # don't move
                return 0, dm
            else:
                rand1 = random.randint(0, 1)
                if rand1 == 0:
                    return 1, mr
                else:
                    return 0, dm
        elif ml > mr:
            if ml > dm:
                # move left
                return -1, ml
            elif ml < dm:
                # don't move
                return 0, dm
            else:
                rand1 = random.randint(0, 1)
                if rand1 == 0:
                    return -1, ml
                else:
                    return 0, dm

        # mr and ml are the same
        elif dm > mr:
            return 0, dm
        elif dm < mr:
            rand1 = random.randint(0, 1)
            if rand1 == 0:
                return 1, mr
            else:
                return -1, ml
        else:
            rand1 = random.randint(0, 2)
            if rand1 == 0:
                return -1, ml
            elif rand1 == 1:
                return 1, mr
            elif rand1 == 2:
                return 0, dm

    # ps = [.33333, 0.5, .6]
    # A
    # x = np.arange(0, nA + 1)
    PDFA1 = conc1PDFs[nA]
    PDFA2 = conc2PDFs[nA]
    PDFA3 = conc3PDFs[nA]

    # B
    # x = np.arange(0, nB + 1)
    PDFB1 = conc1PDFs[nB]
    PDFB2 = conc2PDFs[nB]
    PDFB3 = conc3PDFs[nB]

    allocs = []
    rewards = []
    allChance = 0.0
    for i in range(len(PDFA1)):
        allocs.append([])
        rewards.append([])
        for j in range(len(PDFB1)):
            alloc, reward = calcReward(PDFA1[i], PDFA2[i], PDFA3[i], PDFB1[j], PDFB2[j], PDFB3[j])
            allocs[i].append(alloc)
            chance = ((1 / 3) * (PDFA1[i] + PDFA2[i] + PDFA3[i])) * ((1 / 3) * (PDFB1[j] + PDFB2[j] + PDFB3[j]))
            allChance += chance
            rewards[i].append(reward * chance)

    return allocs, rewards

def findMax(n, diff, setAlloc, conc1PDFs, conc2PDFs, conc3PDFs, nA=0, nB=0):
    max = 0
    maxAlloc = [-1 - 1]
    rewardSums = []
    maxAllots = []
    if setAlloc:
        allocs, rewards = findAllocationAB(nA, nB, diff, conc1PDFs, conc2PDFs, conc3PDFs)
        return sumRewards(rewards), [nA, nB], -1, allocs

    for i in range(0, n + 1):
        allocs, rewards = findAllocationAB(i, n - i, diff, conc1PDFs, conc2PDFs, conc3PDFs)
        rewSum = sumRewards(rewards)
        if rewSum > max:
            max = rewSum
            maxAlloc = [i, n - i]
            maxAllots = allocs
    return max, maxAlloc, -1, maxAllots
    """
    if not equiv:
        if diff > 0:
            allocs, rewards = findAllocationAB(0, n, diff)
            rewSum = sumRewards(rewards)
            rewardSums.append(rewSum)
            max = rewSum
            maxAlloc = [0, n]
            maxAllots = allocs
            return max, maxAlloc, rewardSums, maxAllots
        elif diff < 0:
            allocs, rewards = findAllocationAB(n, 0, diff)
            rewSum = sumRewards(rewards)
            rewardSums.append(rewSum)
            max = rewSum
            maxAlloc = [n, 0]
            maxAllots = allocs
            return max, maxAlloc, rewardSums, maxAllots
        else:
            equalRec = math.floor(n / 2)
            allocs, rewards = findAllocationAB(equalRec, equalRec, diff)
            return sumRewards(rewards), [equalRec, equalRec], [sumRewards(rewards)], allocs
    else:
        equalRec = math.floor(n / 2)
        allocs, rewards = findAllocationAB(equalRec, equalRec, diff)
        return sumRewards(rewards), [equalRec, equalRec], [sumRewards(rewards)], allocs
    """

def transitionProbs(nA, nB, Alloc, Delta, conc1PDFs, conc2PDFs, conc3PDFs):
    # probability of being in location loc and mov m
    PLocMov = []

    # new Delta given a location loc and mov m (Delta + (A-B) ) (indexed the same as PLocMov)
    newDelta = []

    # new Delta as a plus minus on delta (A-B)
    newDeltaAroundDelta = []

    # New A and B for a given movement (Left, no-move, Right)
    newA = [1, 3, 2, 3, 2, 1, 2, 1, 3, 1, 3, 2, 3, 2, 1, 2, 1, 3, 1, 3, 2, 3, 2, 1, 2, 1, 3]
    newB = [3, 1, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 3, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 1, 3, 1, 2]
    for i in range(len(newA)):
        newDelta.append(Delta + (newA[i] - newB[i]))
        newDeltaAroundDelta.append(newA[i] - newB[i])

    def findTransitionReward():
        As = [3, 2, 1, 3, 2, 1, 3, 2, 1]
        Bs = [1, 2, 3, 3, 1, 2, 2, 3, 1]
        PDFA1 = conc1PDFs[nA]
        PDFA2 = conc2PDFs[nA]
        PDFA3 = conc3PDFs[nA]

        PDFB1 = conc1PDFs[nB]
        PDFB2 = conc2PDFs[nB]
        PDFB3 = conc3PDFs[nB]
        DeltToNewDeltaRew = {}

        def findProbGivenABab(conA, conB, a, b):
            PAa = 0.0
            if conA == 1:
                PAa = PDFA1[a]
            elif conA == 2:
                PAa = PDFA2[a]
            elif conA == 3:
                PAa = PDFA3[a]

            PBb = 0.0
            if conB == 1:
                PBb = PDFB1[b]
            elif conB == 2:
                PBb = PDFB2[b]
            elif conB == 3:
                PBb = PDFB3[b]
            return PAa * PBb

        for i in range(nA + 1):
            for j in range(nB + 1):
                m = Alloc[i][j]
                for k in range(len(As)):
                    ProbABab = findProbGivenABab(As[k], Bs[k], i, j)
                    newIndex = (k + m) % len(As)
                    Rew = min(As[newIndex] + Delta, Bs[newIndex] - Delta)
                    newDelta = As[newIndex] + Delta - (Bs[newIndex] - Delta)
                    ProbRew = Rew * ProbABab
                    if DeltToNewDeltaRew.get(newDelta) == None:
                        DeltToNewDeltaRew[newDelta] = ProbRew
                    else:
                        DeltToNewDeltaRew[newDelta] += ProbRew
        return DeltToNewDeltaRew

    def findL0R(concA, concB, nA, nB, Alloc):
        if concA == 1:
            PDFA1 = conc1PDFs[nA]
        elif concA == 2:
            PDFA1 = conc2PDFs[nA]
        else:
            PDFA1 = conc3PDFs[nA]

        if concB == 1:
            PDFB1 = conc1PDFs[nB]
        elif concB == 2:
            PDFB1 = conc2PDFs[nB]
        else:
            PDFB1 = conc3PDFs[nB]
        pl = 0.0
        p0 = 0.0
        pR = 0.0
        for i in range(nA + 1):
            for j in range(nB + 1):
                if Alloc[i][j] == -1:
                    pl += PDFA1[i] * PDFB1[j]
                elif Alloc[i][j] == 0:
                    p0 += PDFA1[i] * PDFB1[j]
                elif Alloc[i][j] == 1:
                    pR += PDFA1[i] * PDFB1[j]
        return (1 / 9) * pl, (1 / 9) * p0, (1 / 9) * pR

    A = [3, 2, 1, 3, 2, 1, 3, 2, 1]
    B = [1, 2, 3, 3, 1, 2, 2, 3, 1]
    AAbs = [3, 2, 1]
    BAbs = [1, 2, 3]
    pxy = []
    py = [0.0, 0.0, 0.0]
    DeltaAllNewDeltaProbs = []
    DeltaAllLORRew = []
    DeltaALlNewDeltas = []
    for i in range(len(A)):
        LOR = findL0R(A[i], B[i], nA, nB, Alloc)
        LmovA = AAbs[(AAbs.index(A[i]) - 1) % 3]
        OmovA = AAbs[AAbs.index(A[i])]
        RmovA = AAbs[(AAbs.index(A[i]) + 1) % 3]

        LmovB = BAbs[(BAbs.index(B[i]) - 1) % 3]
        OmovB = BAbs[BAbs.index(B[i])]
        RmovB = BAbs[(BAbs.index(B[i]) + 1) % 3]

        def returnPiecewiseReward(Delta, movA, movB):
            if Delta > 0:
                return min(Delta + movA, movB)
            elif Delta < 0:
                return min(movA, (-1 * Delta) + movB)
            else:
                return min(movA, movB)

        # new delta probabilities
        DeltaAllNewDeltaProbs.append([LOR[0], LOR[1], LOR[2]])
        # reward for moving left, no move or right
        DeltaAllLORRew.append([returnPiecewiseReward(Delta, LmovA, LmovB), returnPiecewiseReward(Delta, OmovA, OmovB),
                               returnPiecewiseReward(Delta, RmovA, RmovB)])
        # ... into state:
        DeltaALlNewDeltas.append([Delta + (LmovA - LmovB), Delta + (OmovA - OmovB), Delta + (RmovA - RmovB)])
        PLocMov += LOR
        pxy.append([LOR[0], LOR[1], LOR[2]])
        py[0] += LOR[0]
        py[1] += LOR[1]
        py[2] += LOR[2]
    newDeltaSort = {}
    newAroundDeltaSort = {}
    for i in range(len(PLocMov)):
        currNewDeltaProb = newDeltaSort.get(newDelta[i])
        if currNewDeltaProb:
            newDeltaSort[newDelta[i]] = currNewDeltaProb + PLocMov[i]
        else:
            newDeltaSort[newDelta[i]] = PLocMov[i]

        currNewAroundDeltaProb = newAroundDeltaSort.get(newDeltaAroundDelta[i])
        if currNewDeltaProb:
            newAroundDeltaSort[newDeltaAroundDelta[i]] = currNewAroundDeltaProb + PLocMov[i]
        else:
            newAroundDeltaSort[newDeltaAroundDelta[i]] = PLocMov[i]

    Isubj = 0.0
    for i in range(len(pxy)):
        for j in range(len(pxy[0])):
            if pxy[i][j] > 0:
                Isubj += pxy[i][j] * math.log2(pxy[i][j] / ((1 / 9) * py[j]))

    IndvLocProbsDelts = [DeltaAllNewDeltaProbs, DeltaAllLORRew, DeltaALlNewDeltas]
    return newDeltaSort, newAroundDeltaSort, Isubj, IndvLocProbsDelts, findTransitionReward()

# Iterated MDP
def findMaxIter(n, diff, setAlloc, conc1PDFs, conc2PDFs, conc3PDFs, nA=0, nB=0, V=[]):
    def findAllocationABIter(nA, nB, diff, conc1PDFs, conc2PDFs, conc3PDFs, V):
        def calcReward(pA1, pA2, pA3, pB1, pB2, pB3, V):
            lenV = len(V)
            def translateV(index):
                if index > (lenV - 1) / 2:
                    index = (lenV - 1) / 2
                if index < -(lenV - 1) / 2:
                    index = -(lenV - 1) / 2
                return V[int(index) + int((lenV - 1) / 2)]

            vectorized_translateV = np.vectorize(translateV)

            pASum = pA1 + pA2 + pA3
            pBSum = pB1 + pB2 + pB3

            PA1b = pA1 / pASum
            PA2b = pA2 / pASum
            PA3b = pA3 / pASum

            PB1l = pB1 / pBSum
            PB2l = pB2 / pBSum
            PB3l = pB3 / pBSum

            probMatrixLeft = [[PA2b * PB2l, PA1b * PB3l, PA3b * PB1l],
                              [PA2b * PB1l, PA1b * PB2l, PA3b * PB3l],
                              [PA2b * PB3l, PA1b * PB1l, PA3b * PB2l]]

            probMatrixDM = [[PA3b * PB1l, PA2b * PB2l, PA1b * PB3l],
                              [PA3b * PB3l, PA2b * PB1l, PA1b * PB2l],
                              [PA3b * PB2l, PA2b * PB3l, PA1b * PB1l]]

            probMatrixRight = [[PA1b * PB3l, PA3b * PB1l, PA2b * PB2l],
                            [PA1b * PB2l, PA3b * PB3l, PA2b * PB1l],
                            [PA1b * PB1l, PA3b * PB2l, PA2b * PB3l]]
            # A/B    3,1         2,2         1,3
            #       3,3         2,1         1,2
            #       3,2         2,3         1,1

            if diff == 0:

                # Delta
                # 2, 0, -2
                # 0, 1, -1
                # 1, -1, 0
                rewardMatrix = np.asarray([[1,2,1],
                                [3,1,1],
                                [2,2,1]])

                stateValueMatrix = np.asarray([[2,0,-2],
                                    [0,1,-1],
                                    [1,-1,0]])

                translatedReward = rewardMatrix + vectorized_translateV(stateValueMatrix)*gamma
                ml = np.sum(probMatrixLeft*translatedReward)
                dm = np.sum(probMatrixDM*translatedReward)
                mr = np.sum(probMatrixRight*translatedReward)

            elif diff == 1:
                # 3, 1, -1
                # 1, 2, 0
                # 2, 0, 1
                rewardMatrix = np.asarray([[1, 2, 2],
                                           [3, 1, 2],
                                           [2, 3, 1]])

                stateValueMatrix = np.asarray([[3, 1, -1],
                                               [1, 2, 0],
                                               [2, 0, 1]])

                translatedReward = rewardMatrix + vectorized_translateV(stateValueMatrix) * gamma
                ml = np.sum(probMatrixLeft * translatedReward)
                dm = np.sum(probMatrixDM * translatedReward)
                mr = np.sum(probMatrixRight * translatedReward)

            elif diff == -1:
                # 1, -1, -3
                # -1, 0, -2
                # 0, -2, -1
                rewardMatrix = np.asarray([[2, 2, 1],
                                           [3, 2, 1],
                                           [3, 2, 1]])

                stateValueMatrix = np.asarray([[1, -1, -3],
                                               [-1, 0, -2],
                                               [0, -2, -1]])

                translatedReward = rewardMatrix + vectorized_translateV(stateValueMatrix) * gamma
                ml = np.sum(probMatrixLeft * translatedReward)
                dm = np.sum(probMatrixDM * translatedReward)
                mr = np.sum(probMatrixRight * translatedReward)

            elif diff >= 2:
                corr = (diff - 2)
                # Delta
                #        4 +(diff -2), 2 +(diff -2), 0 +(diff -2)
                #        2 +(diff -2), 3 +(diff -2), 1 +(diff -2)
                #        3 +(diff -2), 1 +(diff -2), 2 +(diff -2)
                rewardMatrix = np.asarray([[1, 2, 3],
                                           [3, 1, 2],
                                           [2, 3, 1]])

                stateValueMatrix = np.asarray([[4, 2, 0],
                                               [2, 3, 1],
                                               [3, 1, 2]])

                translatedReward = rewardMatrix + vectorized_translateV(stateValueMatrix+corr) * gamma
                ml = np.sum(probMatrixLeft * translatedReward)
                dm = np.sum(probMatrixDM * translatedReward)
                mr = np.sum(probMatrixRight * translatedReward)

            elif diff <= -2:
                corr = diff + 2
                # 0 + (diff + 2), -2, -4
                # -2, -1, -3
                # -1, -3, -2
                rewardMatrix = np.asarray([[3, 2, 1],
                                           [3, 2, 1],
                                           [3, 2, 1]])

                stateValueMatrix = np.asarray([[0, -2, -4],
                                               [-2, -1, -3],
                                               [-1, -3, -2]])

                translatedReward = rewardMatrix + vectorized_translateV(stateValueMatrix+corr) * gamma
                ml = np.sum(probMatrixLeft * translatedReward)
                dm = np.sum(probMatrixDM * translatedReward)
                mr = np.sum(probMatrixRight * translatedReward)

            if mr > ml:
                if mr > dm:
                    # move right
                    return 1, mr
                elif mr < dm:
                    # don't move
                    return 0, dm
                else:
                    rand1 = random.randint(0, 1)
                    if rand1 == 0:
                        return 1, mr
                    else:
                        return 0, dm
            elif ml > mr:
                if ml > dm:
                    # move left
                    return -1, ml
                elif ml < dm:
                    # don't move
                    return 0, dm
                else:
                    rand1 = random.randint(0, 1)
                    if rand1 == 0:
                        return -1, ml
                    else:
                        return 0, dm

            # mr and ml are the same
            elif dm > mr:
                return 0, dm
            elif dm < mr:
                rand1 = random.randint(0, 1)
                if rand1 == 0:
                    return 1, mr
                else:
                    return -1, ml
            else:
                rand1 = random.randint(0, 2)
                if rand1 == 0:
                    return -1, ml
                elif rand1 == 1:
                    return 1, mr
                elif rand1 == 2:
                    return 0, dm

        # ps = [.33333, 0.5, .6]
        # A
        # x = np.arange(0, nA + 1)
        PDFA1 = conc1PDFs[nA]
        PDFA2 = conc2PDFs[nA]
        PDFA3 = conc3PDFs[nA]

        # B
        # x = np.arange(0, nB + 1)
        PDFB1 = conc1PDFs[nB]
        PDFB2 = conc2PDFs[nB]
        PDFB3 = conc3PDFs[nB]

        allocs = []
        rewards = []
        allChance = 0.0
        for i in range(len(PDFA1)):
            allocs.append([])
            rewards.append([])
            for j in range(len(PDFB1)):
                alloc, reward = calcReward(PDFA1[i], PDFA2[i], PDFA3[i], PDFB1[j], PDFB2[j], PDFB3[j], V)
                allocs[i].append(alloc)
                chance = ((1 / 3) * (PDFA1[i] + PDFA2[i] + PDFA3[i])) * ((1 / 3) * (PDFB1[j] + PDFB2[j] + PDFB3[j]))
                allChance += chance
                rewards[i].append(reward * chance)

        return allocs, rewards

    max1 = 0
    maxAlloc = [-1, -1]
    maxAllots = []
    if setAlloc:
        allocs, rewards = findAllocationABIter(nA, nB, diff, conc1PDFs, conc2PDFs, conc3PDFs, V)
        return sumRewards(rewards), [nA, nB], -1, allocs

    for i in range(0, n + 1):
        allocs, rewards = findAllocationABIter(i, n - i, diff, conc1PDFs, conc2PDFs, conc3PDFs, V)
        rewSum = sumRewards(rewards)
        if rewSum > max1:
            max1 = rewSum
            maxAlloc = [i, n - i]
            maxAllots = allocs
    return max1, maxAlloc, -1, maxAllots

def IteratePolicy(n, stateValues, Deltas, MIs, AllocA, AllocB):
    conc1PDFs, conc2PDFs, conc3PDFs = generatePDF(n, [1 / (1 + 2), 2 / (2 + 2), 3 / (3 + 2)])
    #Deltas = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    setAlloc = False
    if AllocA > 0:
        setAlloc = True

    Allocs = []
    nAs = []
    nBs = []
    rewards = []
    nextStateReward = []
    for i in range(len(Deltas)):
        allocsRet = findMaxIter(n, Deltas[i], setAlloc, conc1PDFs, conc2PDFs, conc3PDFs, nA=AllocA, nB=AllocB, V=stateValues)
        Allocs.append(allocsRet[3])
        nAs.append(allocsRet[1][0])
        nBs.append(allocsRet[1][1])
        rewards.append(allocsRet[0])
        nextStateReward.append(findRewardFromAllocs(allocsRet[3], allocsRet[1][0], allocsRet[1][1], Deltas[i], conc1PDFs, conc2PDFs, conc3PDFs)[1])

    newDeltas = []
    subjPerDelta = []
    transProbsPerDelta = []
    ProbDtoD = {}
    RewardsDtoD = {}
    for i in range(len(Deltas)):
        transRet = transitionProbs(nAs[i], nBs[i], Allocs[i], Deltas[i], conc1PDFs, conc2PDFs, conc3PDFs)
        transProbsPerDelta.append(transRet[3])
        newDeltas.append(transRet[0])
        ProbDtoD[Deltas[i]] = transRet[0]
        subjPerDelta.append(transRet[2])
        RewardsDtoD[Deltas[i]] = transRet[4]

    def getallProbs(newDeltaDict, n):
        deltas = np.arange(-n, n + 1)
        deltaProbs = []
        for j in deltas:
            if newDeltaDict.get(j) is None:
                deltaProbs.append(0.0)
            else:
                deltaProbs.append(newDeltaDict[j])
        return deltaProbs

    deltas = np.arange(-5, 5 + 1)
    ps = []
    for newD in range(len(deltas)):
        currPs = getallProbs(newDeltas[newD], 5)
        if sum(currPs) < 1:
            if deltas[newD] < 0:
                currPs[0] += 1 - sum(currPs)
            else:
                currPs[len(currPs)-1] += 1 - sum(currPs)
        ps.append(currPs)

    #for i in range(len(ps)):
    #    print(ps[i])
    #print(len(ps))
    #for j in ps:
    #    print(len(j))
    #print("end")
    q = getSteadyStateDist(ps)
    adaptivePs = [0.00041115, 0.00205625, 0.00912688, 0.05959926, 0.24065731, 0.37907344, 0.24078636, 0.05700681, 0.00890222, 0.0019817,  0.00039861]
    #print(q)

    #plt.plot(deltas, q, label = "Equivalent", zorder = 2, color = 'b')
    #plt.scatter(deltas, q, color='b', zorder = 2)
    #plt.plot(deltas, adaptivePs, label = "Adaptive", zorder = 2, color = 'orange')
    #plt.scatter(deltas, adaptivePs, color = 'orange', zorder = 2)
    #plt.xlabel("$\Delta$")
    #plt.ylabel("$\Pi$")
    #plt.legend()
    #plt.grid(zorder = 1)
    #plt.show()
    #exit()

    deltHigh = Deltas[len(Deltas) - 1]
    ProbDtoDnp = np.zeros([len(Deltas), len(Deltas)])
    RewardsDtoDnp = np.zeros([len(Deltas), len(Deltas)])
    for i in range(len(ProbDtoDnp)):
        indexi = i - deltHigh
        for j in range(len(ProbDtoDnp[0])):
            indexj = j - deltHigh
            if ProbDtoD.get(indexi) is not None:
                ProbDtoDsmall = ProbDtoD.get(indexi)
                if ProbDtoDsmall.get(indexj) is not None:
                    ProbDtoDnp[i][j] = ProbDtoD[indexi][indexj]
            if RewardsDtoD.get(indexi) is not None:
                RewardsDtoDsmall = RewardsDtoD.get(indexi)
                if RewardsDtoDsmall.get(indexj) is not None:
                    RewardsDtoDnp[i][j] = RewardsDtoD[indexi][indexj]

    def calculate_value_function(P, R, gamma):
        r = np.sum(P * R, axis=1)
        I = np.eye(P.shape[0])
        V = np.linalg.inv(I - gamma * P) @ r
        return V

    stateValues = calculate_value_function(ProbDtoDnp, RewardsDtoDnp, gamma)
    allTrans = []
    front_elements = Deltas[:2] - 2
    back_elements = Deltas[-2:] + 2

    MIDelta = []
    for i in range(len(nAs)):
        MIDelta.append(MIs[nAs[i]] + MIs[nBs[i]])

    FUllDeltas = np.concatenate((front_elements, Deltas, back_elements))
    for i in range(len(newDeltas)):
        currTrans = []
        for j in range(len(FUllDeltas)):
            prob = newDeltas[i].get(FUllDeltas[j])
            if prob:
                currTrans.append(prob)
            else:
                currTrans.append(0.0)
        # """ fixes trailing and leading references to outside deltas
        if currTrans[0] > 0:
            currTrans[2] = currTrans[2] + currTrans[0]

        if currTrans[1] > 0:
            currTrans[2] = currTrans[2] + currTrans[1]
        currTransLen = len(currTrans)
        if currTrans[currTransLen - 1] > 0:
            currTrans[currTransLen - 3] = currTrans[currTransLen - 3] + currTrans[currTransLen - 1]

        if currTrans[currTransLen - 2] > 0:
            currTrans[currTransLen - 3] = currTrans[currTransLen - 3] + currTrans[currTransLen - 2]

        currTrans = currTrans[2:currTransLen - 2]
        # """
        allTrans.append(currTrans)
    rewards = []
    for i in range(len(Deltas)):
        allocsRet = findMaxIter(n, Deltas[i], setAlloc, conc1PDFs, conc2PDFs, conc3PDFs, nA=AllocA, nB=AllocB, V=stateValues)
        rewards.append(allocsRet[0])

    q = getSteadyStateDist(allTrans)
    currAverageReward = sum(np.asarray(q) * np.asarray(rewards))
    # currAverageReward = sum(np.asarray(rewards)) / len(rewards)
    currSubjMI = sum(np.asarray(q) * np.asarray(subjPerDelta))
    currAverageMI = sum(np.asarray(q) * np.asarray(MIDelta))
    return q, currAverageReward, currAverageMI, currSubjMI, transProbsPerDelta, stateValues
# -----------------

#deltas = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
maxDelta = 5
deltas = np.arange(-maxDelta, maxDelta + 1)
D = len(deltas)
V = []
for i in range(D):
    V.append(0.0)

def iterateMDP(Recs, StateValues, deltas, MIs, AllocA = -1, AllocB = -1):
    arrs = []
    Iterret = 0
    firstIter = []
    firstIterFlag = False
    for i in range(100):
        #print(i)
        oldStateValues = StateValues
        Iterret = IteratePolicy(Recs, StateValues, deltas, MIs, AllocA, AllocB)
        if not firstIterFlag:
            firstIterFlag = True
            firstIter = Iterret
        StateValues = Iterret[5]
        arrs.append(StateValues)
        flag = False
        for i in range(len(oldStateValues)):
            if oldStateValues[i] != StateValues[i]:
                flag = True

        if not flag:
            print("converged")
            #print(Iterret[5])
            break
        #print(Iterret[5])
    return arrs, Iterret[1], Iterret[2], firstIter

def getMIsforn(n):
    conc1PDFs, conc2PDFs, conc3PDFs = generatePDF(n, [1 / (1 + 2), 2 / (2 + 2), 3 / (3 + 2)])
    MIs = CaluclateMIs(conc1PDFs, conc2PDFs, conc3PDFs, n)
    return MIs
maxRec = 32
Recs = [i for i in range(2, maxRec+1) if i % 4 == 0]
#Recs = range(0, maxRec+1)

MIs = []
arrs_grow = []
arrs_MI = []
arrs_grow_First = []
arrs_MI_First = []

V = []
for j in range(D):
    V.append(0.0)
deltas = np.arange(-maxDelta, maxDelta + 1)
MIs = getMIsforn(100)
#iterateMDP(100, V, deltas, MIs, AllocA=50, AllocB=50)

for i in range(len(Recs)):
    print(Recs[i])
    D = len(deltas)
    V = []
    for j in range(D):
        V.append(0.0)
    MIs = getMIsforn(Recs[i])
    iter = iterateMDP(Recs[i], V, deltas, MIs)
    arrs_grow.append(iter[1])
    arrs_MI.append(iter[2])
    arrs_grow_First.append(iter[3][1])
    arrs_MI_First.append(iter[3][2])


#recAll = 20
#MIs = getMIsforn(recAll)
#arrs_grow_All = []
#arrs_MI_All = []
#for i in range(recAll):
#    D = len(deltas)
#    V = []
#    for j in range(D):
#        V.append(0.0)
#    allA = i
#    allB = recAll - allA
#    iter = iterateMDP(recAll, V, deltas, MIs, AllocA=allA, AllocB=allB)
#    arrs_grow_All.append(iter[1])
#    arrs_MI_All.append(iter[2])


arrs_grow_equiv = []
arrs_MI_equiv = []
arrs_grow_First_equiv = []
arrs_MI_First_equiv = []
for i in range(len(Recs)):
    D = len(deltas)
    V = []
    for j in range(D):
        V.append(0.0)
    print(Recs[i])
    allA = int(Recs[i]/2)
    allB = Recs[i] - allA
    MIs = getMIsforn(Recs[i])
    iter = iterateMDP(Recs[i], V, deltas, MIs, AllocA = allA, AllocB = allB)
    arrs_grow_equiv.append(iter[1])
    arrs_MI_equiv.append(iter[2])
    arrs_grow_First_equiv.append(iter[3][1])
    arrs_MI_First_equiv.append(iter[3][2])




plt.plot(Recs, arrs_grow, label = "$\hat{R}^*_{Adaptive}$", zorder = 2)
plt.plot(Recs, arrs_grow_equiv, label = "$\hat{R}^*_{Equivalent}$", zorder = 2)
plt.plot(Recs, np.asarray(arrs_grow_First), label = "$\hat{R}_{Adaptive}$", zorder = 2)
plt.plot(Recs, np.asarray(arrs_grow_First_equiv), label = "$\hat{R}_{Equivalent}$", zorder = 2)
plt.scatter(Recs, arrs_grow, zorder = 2)
plt.scatter(Recs, arrs_grow_equiv, zorder = 2)
plt.scatter(Recs, np.asarray(arrs_grow_First), zorder = 2)
plt.scatter(Recs, np.asarray(arrs_grow_First_equiv), zorder = 2)
plt.xlabel("Receptor Count $N_{tot}$")
plt.ylabel("Expected Reward $\hat{R}_{Strategy}$")
#plt.scatter(Recs, arrs_grow_All, label = "Set Allocations A/B", zorder = 2)
plt.grid(zorder = 1)
plt.legend()
plt.show()

plt.plot(np.asarray(arrs_grow) * -1, arrs_MI, label = "Policy Iterated Approach Adapt", zorder = 2)
plt.plot(np.array(arrs_grow_equiv) * -1, arrs_MI_equiv, label = "Policy Iterated Approach Equiv", zorder = 2)
plt.plot(np.array(arrs_grow_First) * -1, arrs_MI_First, label = "Steady State Adapt", zorder = 2)
plt.plot(np.asarray(arrs_grow_First_equiv) * -1, arrs_MI_First_equiv, label = "Steady State Equiv", zorder = 2)
plt.scatter(np.asarray(arrs_grow) * -1, arrs_MI, zorder = 2)
plt.scatter(np.array(arrs_grow_equiv) * -1, arrs_MI_equiv, zorder = 2)
plt.scatter(np.asarray(arrs_grow_First) * -1, arrs_MI_First, zorder = 2)
plt.scatter(np.asarray(arrs_grow_First_equiv) * -1, arrs_MI_First_equiv, zorder = 2)
plt.xlabel("Distortion (Negative Expected Reward)")
plt.ylabel("Mutual Information $I_{synt}$ [bits]")
#plt.scatter(Recs, arrs_grow_All, label = "Set Allocations A/B", zorder = 2)
plt.grid(zorder = 1)
plt.legend()
plt.show()
