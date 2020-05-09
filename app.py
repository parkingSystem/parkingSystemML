# import numpy as np
import os
import sys
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import requests
import matplotlib.image as mpimg
import tensorflow as tf

graph = tf.get_default_graph()

NOMEROFF_NET_DIR = os.path.abspath('./nomeroff-net-master')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

sys.path.append(NOMEROFF_NET_DIR)

from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync

nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")

rectDetector = RectDetector()
optionsDetector = OptionsDetector()
optionsDetector.load("latest")
textDetector = TextDetector.get_static_module("kz")()
textDetector.load("latest")

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    global graph
    if request.method == "POST":
        file = request.files["file"]
        if file and (file.content_type.rsplit('/', 1)[1] in ALLOWED_EXTENSIONS).__bool__():
            filename = secure_filename(file.filename)
            file.save('images/' + filename)
            imgPath = ('images/' +  filename)
            img = mpimg.imread(imgPath)
            with graph.as_default():
                NP = nnet.detect([img])
                cv_img_masks = filters.cv_img_mask(NP)
                arrPoints = rectDetector.detect(cv_img_masks)
                zones = rectDetector.get_cv_zonesBGR(img, arrPoints)
                regionIds, stateIds, countLines = optionsDetector.predict(zones)
                regionNames = optionsDetector.getRegionLabels(regionIds)
                textArr = textDetector.predict(zones)
                textArr = textPostprocessing(textArr, regionNames)
                print(textArr)
                response = requests.post('https://ab1b27f2.ngrok.io/api/validateOut', data={'carNumber':textArr,'parkId':'1'})
                return response.json()
    return jsonify({'error': ''})
    # if 'file' not in request.files:
    #     return jsonify({'error': ''})
    # file = request.files['file']
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     # file.save(os.path.join(app.config[IMG_PATH], filename))
    #     file.save(IMG_PATH + filename)
    #     text_array = predict(IMG_PATH + filename)
    #     return jsonify(text_array)
    # return jsonify({'error': ''})

if __name__ == "__main__":
    app.debug = False
    app.run(host='127.0.0.1', port=3000, threaded=False)