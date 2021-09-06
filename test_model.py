import random
from collections import deque
import time
import cv2
import numpy as np
from statistics import mean

from models import inception_v3
from utilities.grab_screen import grabScreen
from utilities.motion import motionDetection
from utilities.get_keys import keyCheck
from utilities.direct_keys import pressKey, releaseKey, W, A, S, D

gameWidth = 1920
gameHeight = 1080
howFarRemove = 800
rs = (20, 15)
logLen = 25
motionReq = 800
motionLog = deque(maxlen = logLen)
width = 480
height = 270
learningRate = 1e-3
nbOfEpochs = 10
choices = deque([], maxlen=5)
hlHistory = 250
choiceHistory = deque([], maxlen=hlHistory)
t_time = 0.25

def straight():
    pressKey(W)
    releaseKey(A)
    releaseKey(D)
    releaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        pressKey(W)
    else:
        releaseKey(W)
    pressKey(A)
    releaseKey(S)
    releaseKey(D)


def right():
    if random.randrange(0, 3) == 1:
        pressKey(W)
    else:
        releaseKey(W)
    pressKey(D)
    releaseKey(A)
    releaseKey(S)


def reverse():
    pressKey(S)
    releaseKey(A)
    releaseKey(W)
    releaseKey(D)


def forwardLeft():
    pressKey(W)
    pressKey(A)
    releaseKey(D)
    releaseKey(S)


def forwardRight():
    pressKey(W)
    pressKey(D)
    releaseKey(A)
    releaseKey(S)


def reverseLeft():
    pressKey(S)
    pressKey(A)
    releaseKey(W)
    releaseKey(D)


def reverseRight():
    pressKey(S)
    pressKey(D)
    releaseKey(W)
    releaseKey(A)


def noKeys():
    if random.randrange(0, 3) == 1:
        pressKey(W)
    else:
        releaseKey(W)
    releaseKey(A)
    releaseKey(S)
    releaseKey(D)


model = inception_v3(width, height, 3, learningRate, 9)
modelName = ''
model.load(modelName)

def main():
    for i in range(4, 0, -1):
        print(i)
        time.sleep(1)

    isPaused = False
    modeChoice = 0

    screen = grabScreen(region=(0, 40, gameWidth, gameHeight + 40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (width, height))

    tMinus, tNow, tPlus = prev, prev, prev

    while True:
        if not paused:
            screen = grabScreen(region = (0, 40, gameWidth, gameHeight + 40))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            screen = cv2.resize(screen, (width, height))

            deltaCountLast = motionDetection(screen, tMinus, tPlus)

            tMinus = tNow
            tNow = tPlus
            tPlus = screen
            tPlus = cv2.blur(tPlus, (4, 4))

            prediction = model.predict([screen.reshape(width, height, 3)])[0]
            prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1, 1.8, 1.8, 0.5, 0.5, 0.2])
            modeChoice = np.argmax(prediction)


            if modeChoice == 0:
                straight()
            elif modeChoice == 1:
                reverse()
            elif modeChoice == 2:
                left()
            elif modeChoice == 3:
                right()
            elif modeChoice == 4:
                forwardLeft()
            elif modeChoice == 5:
                forwardRight()
            elif modeChoice == 6:
                reverseLeft()
            elif modeChoice == 7:
                reverseRight()
            elif modeChoice == 8:
                noKeys()

            motionLog.append(deltaCountLast)
            motion_avg = round(mean(motionLog), 3)

            if motion_avg < motionReq and len(motionLog) >= logLen:
                print('WERE PROBABLY STUCK FFS, initiating some evasive maneuvers.')

                # 0 = reverse straight, turn left out
                # 1 = reverse straight, turn right out
                # 2 = reverse left, turn right out
                # 3 = reverse right, turn left out

                quick_choice = random.randrange(0, 4)

                if quick_choice == 0:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forwardLeft()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 1:
                    reverse()
                    time.sleep(random.uniform(1, 2))
                    forwardRight()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 2:
                    reverseLeft()
                    time.sleep(random.uniform(1, 2))
                    forwardRight()
                    time.sleep(random.uniform(1, 2))

                elif quick_choice == 3:
                    reverseRight()
                    time.sleep(random.uniform(1, 2))
                    forwardLeft()
                    time.sleep(random.uniform(1, 2))

                for i in range(logLen - 2):
                    del motionLog[0]

        keys = keyCheck()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                releaseKey(A)
                releaseKey(W)
                releaseKey(D)
                time.sleep(1)

if __name__ == "__main__":
    main()