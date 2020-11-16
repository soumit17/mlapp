from sklearn.externals import joblib
from flask import Flask, request
import numpy as np
import flasgger
from flasgger import Swagger
import tensorflow.keras
from PIL import Image, ImageOps
import io
import base64
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

app=Flask(__name__)
Swagger(app)

model_insurance=joblib.load('/home/ubuntu/mlapp/insurance.ml')
# model_insurance=joblib.load('insurance.ml')

np.set_printoptions(suppress=True)
model_cardam = tensorflow.keras.models.load_model('/home/ubuntu/mlapp/keras_model.h5')
# model_cardam = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

detect_fn = tf.saved_model.load('/home/ubuntu/mlapp/saved_model')
PATH_TO_LABELS=r"/home/ubuntu/mlapp/saved_model/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def B64argtoCV2(argname):
    inputdata = request.form[argname]
    if str(inputdata)[:2] == "b'":
        im_bytes = base64.b64decode(bytes(str(inputdata)[2:], 'ascii'))
    else:
        im_bytes=base64.b64decode(inputdata)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

def CV2toB64(image):
    _, im_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes)                                                             

@app.route('/insurance_cost')
def predict():
    """Check insurance cost for a person
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: gender
        in: query
        type: number
        required: true
      - name: bmi
        in: query
        type: number
        required: true
      - name: children
        in: query
        type: number
        required: true
      - name: smoker
        in: query
        type: number
        required: true
      - name: region
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output cost
        
    """
    insurance_cost=model_insurance.predict([[int(request.args['age']),
                                int(request.args['gender']),
                                int(request.args['bmi']),
                                int(request.args['children']),
                                int(request.args['smoker']),
                                int(request.args['region'])]])
    return str(round(insurance_cost[0],2))
    
@app.route('/car_damage_B24',methods=["POST"])
def predict_car_damage_B24():
    """Check car is damaged or not
    This is using docstrings for specifications.
    ---
    parameters:
      - name: image
        in: formData
        type: text
        required: true
      
    responses:
        200:
            description: Output result
        
    """
    inputdata = request.form['image']
    imgdata=base64.b64decode(bytes(str(inputdata)[2:], 'ascii'))
    im = Image.open(io.BytesIO(imgdata))
    size = (224, 224)
    image = ImageOps.fit(im, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model_cardam.predict(data)
    return str('Car is Damaged!' if prediction[0][0]>.5 else 'Car is Good!')
    
@app.route('/car_damage',methods=["POST"])
def predict_car_damage():
    """Check car is damaged or not
    This is using docstrings for specifications.
    ---
    parameters:
      - name: imagefile
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: Output result
        
    """
    image = Image.open(request.files.get("imagefile"))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model_cardam.predict(data)
    return str('Car is Damaged!' if prediction[0][0]>.5 else 'Car is Good!')
    
def garbage():
    """Identify litter via object detection
    This is using docstrings for specifications.
    ---
    parameters:
      - name: imageB64
        in: formData
        type: text
        required: true
      
    responses:
        200:
            description: Output result
        
    """
    image = B64argtoCV2('imageB64')
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_with_detections = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.6,
      agnostic_mode=False)
    return CV2toB64(image_with_detections)
    
if __name__=='__main__':
    app.run()
