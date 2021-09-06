import os
import time
import cv2
import numpy as np

from utilities.grab_screen import grabScreen
from utilities.get_keys import keyCheck

keyMap = {
    "W": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "S": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "A": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "D": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "WS": [0, 0, 0, 0, 1, 0, 0, 0, 0],
    "WD": [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "SA": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "SD": [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "NK": [0, 0, 0, 0, 0, 0, 0, 0, 1],
    "default": [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

startingValue = 1058
while True:
    fileName = f"data/trainingData-{startingValue}.npy"
    if os.path.isfile(fileName):
        startingValue += 1
    else:
        break

def keysToOutput(keys):
    if "".join(keys) in keyMap:
        return keyMap["".join(keys)]
    return keyMap["default"]

def main(fileName, startingValue):
    trainingData = []
    for i in range(4, 0, -1):
        print(i)
        time.sleep(1)
    
    isPaused = False
    while True:
        if not isPaused:
            screen = grabScreen(region=(0, 40, 1920, 1120))
            screen = cv2.resize(screen, (480, 270))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            keys = keyCheck()
            output = keysToOutput(keys)
            trainingData.append([screen, output])
            if len(trainingData) % 100 == 0:
                if len(trainingData) == 500:
                    np.save(fileName, trainingData)
                    trainingData = []
                    startingValue += 1
                    fileName = os.path.join(os.getcwd(), f"data/trainingData-{startingValue}.npy")
        
        keys = keyCheck()
        if "T" in keys:
            if isPaused:
                isPaused = False
                time.sleep(1)
            else:
                isPaused = True
                time.sleep(1)

if __name__ == "__main__":
    main(fileName, startingValue)