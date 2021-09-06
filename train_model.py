import os
from numpy import np
from random import shuffle

from models import alexnet

fileIndexEnd = 1860
width = 480
height = 270
learningRate = 1e-3
nbOfEpochs = 30
modelName = ""
prevModel = ""
aModelIsLoaded = True

model = alexnet(width, height, 3, learningRate, 9, modelName)

if aModelIsLoaded:
    model.load(prevModel)

for e in range(nbOfEpochs):
    dataOrder = [i for i in range(1, fileIndexEnd + 1)]
    shuffle(dataOrder)
    for count, i in enumerate(dataOrder):
        try:
            fileName = os.path.join(os.getcwd(), f"data/trainingData-{i}.npy")
            trainData = np.load(fileName)
            train = trainData[:-50]
            test = trainData[-50:]
            X = np.array([i[0] for i in train]).reshape(-1, width, height, 3)
            Y = [i[1] for i in train]
            test_x = np.array([i[0] for i in test]).reshape(-1, width, height, 3)
            test_y = [i[1] for i in test]
            model.fit({'input': X}, 
                      {'targets': Y}, 
                      n_epoch=1, 
                      validation_set=({'input': test_x}, {'targets': test_y}),
                      snapshot_step=2500, 
                      show_metric=True, 
                      run_id=modelName)
            if count % 10 == 0:
                model.save(modelName)
        except Exception as e:
            print(e)