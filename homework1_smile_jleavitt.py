import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time

def fPC (y, yhat):
    #Return the average of time predictions were accurate over all images
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    total = np.zeros(len(X))
    for p in predictors:
        r1,c1,r2,c2 = p
        
        #get difference between pixel values for all images
        diff = X[:, r1, c1] - X[:,r2,c2]
        #Convert to 1 if value is positive else 0
        boolDiff =  np.multiply(diff > 0, 1)
        total += boolDiff
        
    #Convert ensemble prediction to bool 0 or 1
    yhat = np.multiply(total/len(predictors) > 0.5, 1)
    
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
                        testPreds = predictors.copy()
                        testPreds.append(pix)
                        predictorAccuracy = measureAccuracyOfPredictors(testPreds, trainingFaces, trainingLabels)

                        if predictorAccuracy > bestPredictor:
                            bestPredictor = predictorAccuracy
                            bestLocation =  pix
        predictors.append(bestLocation)

    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        num_pred = 1
        for p in predictors:
            r1, c1, r2, c2 = p
            # Show r1,c1
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/2.0
            ax.annotate(str(num_pred), (cx,cy), color='red', fontsize=8, ha='center', va='center')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/2.0
            ax.annotate(str(num_pred), (cx,cy), color='blue', fontsize=8, ha='center', va='center')
            ax.add_patch(rect)
            num_pred += 1
        # Display the merged result
        plt.show()

    return predictors


def loadData (which):
    faces = np.load(f"{which}ingFaces.npy")
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load(f"{which}ingLabels.npy")
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    run = [400, 600, 800, 1000, 1200, 1600, 2000]
    print(f"[n]  |Training Accuracy|Test Accuracy|Time to execute (seconds)")
    for r in run:
        startTime = time.time()
        sampleLabels = trainingLabels[:r]
        sampleFaces = trainingFaces[:r]
        finalPredictors = stepwiseRegression(sampleFaces, sampleLabels, testingFaces, testingLabels)

        trainAcc = measureAccuracyOfPredictors(finalPredictors, sampleFaces, sampleLabels)
        testAcc = measureAccuracyOfPredictors(finalPredictors, testingFaces, testingLabels)

        #Note: Time to execute includes time that the predictor image was open
        print(f"{r}| {trainAcc}         | {testAcc}  | {time.time()-startTime}")