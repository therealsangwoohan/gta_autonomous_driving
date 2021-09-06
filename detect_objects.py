import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utilities.grab_screen import grabScreen
import cv2
from utils import label_map_util
from utils import visualization_utils


modelName = 'ssd_mobilenet_v1_coco_11_06_2017'
modelFile = modelName + '.tar.gz'
downloadBase = 'http://download.tensorflow.org/models/object_detection/'
pathToCKPT = modelName + '/frozen_inference_graph.pb'
pathToLabels = os.path.join('data', 'mscoco_label_map.pbtxt')
numClasses = 90

# Download model.
opener = urllib.request.URLopener()
opener.retrieve(downloadBase + modelFile, modelFile)
modelTarFile = tarfile.open(modelFile)
for file in modelTarFile.getmembers():
  fileName = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in fileName:
    modelTarFile.extract(file, os.getcwd())


# Load model.
detectionGraph = tf.Graph()
with detectionGraph.as_default():
  odGraphDef = tf.GraphDef()
  with tf.gfile.GFile(pathToCKPT, 'rb') as fid:
    serializedGraph = fid.read()
    odGraphDef.ParseFromString(serializedGraph)
    tf.import_graph_def(odGraphDef, name='')


# Load map.
labelMap = label_map_util.load_labelmap(pathToLabels)
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=numClasses, use_display_name=True)
categoryIndex = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (imWidth, imHeight) = image.size
  return np.array(image.getdata()).reshape(
      (imHeight, imWidth, 3)).astype(np.uint8)


imageSize = (12, 8)
with detectionGraph.as_default():
  with tf.Session(graph=detectionGraph) as sess:
    while True:
      screen = cv2.resize(grabScreen(region=(0,40,1280,745)), (800,450))
      imageNP = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      imageNPExpanded = np.expand_dims(imageNP, axis=0)
      imageTensor = detectionGraph.get_tensor_by_name('image_tensor:0')
      boxes = detectionGraph.get_tensor_by_name('detection_boxes:0')
      scores = detectionGraph.get_tensor_by_name('detection_scores:0')
      classes = detectionGraph.get_tensor_by_name('detection_classes:0')
      numDetections = detectionGraph.get_tensor_by_name('num_detections:0')
      (boxes, scores, classes, numDetections) = sess.run(
          [boxes, scores, classes, numDetections],
          feed_dict={imageTensor: imageNPExpanded})
      visualization_utils.visualize_boxes_and_labels_on_image_array(
          imageNP,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          categoryIndex,
          use_normalized_coordinates=True,
          line_thickness=8)
      cv2.imshow('window',imageNP)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break