import cv2

def deltaImages(tMinus, tPlus):
    return cv2.absdiff(tPlus, tMinus)

def motionDetection(screen, tMinus, tPlus):
    deltaView = deltaImages(tMinus, tPlus)
    deltaView = cv2.threshold(deltaView, 16, 255, 3)[1]
    cv2.normalize(deltaView, deltaView, 0, 255, cv2.NORM_MINMAX)
    imgCountView = cv2.cvtColor(deltaView, cv2.COLOR_RGB2GRAY)
    deltaCount = cv2.countNonZero(imgCountView)
    cv2.addWeighted(screen, 1.0, deltaCount, 0.6,0)
    return deltaCount