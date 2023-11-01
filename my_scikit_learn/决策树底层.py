import pickle
import operator
from math import log
from matplotlib.pyplot import plt
from matplotlib.font_manager import FontProperties


def createDataset():
    Dataset=[
        [0,0,0,0,'no'],
        [0,0,0,1,'yes'],
        [0,1,0,1,'yes'],
        [0,1,1,0,'no'],
        [0,0,0,0,'no'],

        [1,0,0,0,'no'],
        [1,0,0,1,'no'],
        [1,1,1,1,'yes'],
        [1,0,1,2,'yes'],
        [1,0,1,2,'yes'],

        [2,0,1,2,'yes'],
        [2,0,1,1,'yes'],
        [2,1,0,1,'yes'],
        [2,1,0,2,'yes'],
        [2,0,0,0,'no']]
    labels = ['F1-AGE','F2-WORK','F3-HOME','F4-LOAN']
    return Dataset,labels

def createTree(dataset,labels,featLables):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]==len(classList)):
        return classList[0]
    if len(dataset[0])==1:
        print("遍历完")
        return majorityCnt(classList)
    bestFeat = chooseBestFeatToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    featLables.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataset,bestFeat,value),labels)
    

    return myTree
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote]+=1
    classCounted = sorted(classCount.items(),key=operator.itemgetter,reverse=True)
    return classCounted[0][0]
def chooseBestFeatToSplit(dataset):
    numFeatures = len(dataset[0])-1
    baseEntropy = calcShannonEnt(dataset)
     


def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)

def calcShannonEnt(dataset) :
    numexamples = len( dataset)
    labelCounts = {}
    for featVec in dataset :
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts [currentlabel] =0
        labelCounts [currentlabel] += 1
    shannonEnt =0
    for key in labelCounts :















