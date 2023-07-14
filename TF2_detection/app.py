from flask import Flask,render_template,request

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import warnings
from pedestrain_detector import PedestrianDetector
warnings.filterwarnings('ignore') # Suppress Matplotlib warnings   

app = Flask(__name__,template_folder='./templates',static_folder='./static')
app.config['SECRET_KEY'] = 'the random string'

@app.route('/')
def home():
    return render_template('index.html',img_path='static/others/home.png')

@app.route('/',methods=["GET","POST"])
def detect():
    if request.method=="POST":
        if 'picture' in request.files:
            f=request.files['picture']
            filename=f.filename
            print(filename)
            try:
              basepath = os.path.dirname(__file__)
              filepath = os.path.join(basepath,'uploads',filename)
              f.save(filepath)
              detector=PedestrianDetector(path_to_model='./exported-models/my-faster-rcnn/saved_model',
                                          path_to_label_map='./exported-models/my-faster-rcnn/label_map.pbtxt')
              detection_img=detector.detect_from_img(filepath)
              print('Successfully detected')
              detected_img_path = os.path.join("static/detections/" , filename)             
              cv2.imwrite(detected_img_path,detection_img)
              return render_template('index.html',img_path=detected_img_path)
            except ValueError:
               return render_template('index.html',img_path='static/others/ERROR.png')

app.run()
            
