import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time

def fPC (y, yhat):
    #Return the average of time the prediction was accurate 
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    total = np.zeros(len(X))
    for p in predictors:
        r1,c1,r2,c2 = p
        
        #get difference
        diff = X[:, r1, c1] - X[:,r2,c2]
        #Convert to bool 0 or 1
        boolDiff = 1 * (diff > 0)
        total += boolDiff
    #Find prediction
    yhat = 1 * (total/len(predictors) > 0.5)
    return fPC(y, yhat)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):

    imageSize = 24
    predictors = []

    for i in range(6): #get 6 predictors
        bestPredictor = 0
        bestLocation = 0
        for r1 in range(imageSize):
            for c1 in range(imageSize):
                for r2 in range(imageSize):
                    for c2 in range(imageSize):

                        pix = (r1,c1,r2,c2)

                        if r1 == r2 and c1 == c2 or pix in predictors:
                            #Same pixel or already in predictors
                            continue

                        predictorAccuracy = measureAccuracyOfPredictors(predictors + list((pix,)), trainingFaces, trainingLabels)

                        if predictorAccuracy > bestPredictor:
                            bestPredictor = predictorAccuracy
                            bestLocation =  pix
        predictors.append(bestLocation)

    return predictors


def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    run = [400, 600, 800, 1000, 1200, 1600, 2000]
    print(f"[n]  |Training Accuracy|Test Accuracy|Time to execute (seconds)")
    for r in run:
        startTime = time.time()
        sampleLabels = trainingLabels[:r]
        sampleFaces = trainingFaces[:r, :, :]
        finalPredictors = stepwiseRegression(sampleFaces, sampleLabels, testingFaces, testingLabels)

        trainAcc = measureAccuracyOfPredictors(finalPredictors, sampleFaces, sampleLabels)
        testAcc = measureAccuracyOfPredictors(finalPredictors, testingFaces, testingLabels)

        print(f"{r}| {trainAcc}         | {testAcc}  | {time.time()-startTime}")