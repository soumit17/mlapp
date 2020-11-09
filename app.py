from sklearn.externals import joblib
from flask import Flask, request
import numpy as np
import flasgger
from flasgger import Swagger
import tensorflow.keras
from PIL import Image, ImageOps

app=Flask(__name__)
Swagger(app)

model_insurance=joblib.load('/home/ubuntu/mlapp/insurance.ml')
# model_insurance=joblib.load('insurance.ml')

np.set_printoptions(suppress=True)
model_cardam = tensorflow.keras.models.load_model('/home/ubuntu/mlapp/keras_model.h5')
# model_cardam = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

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
    
@app.route('/car_damage',methods=["POST"])
def predict_car_damage():
    """Check car is damaged or not
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: Output result
        
    """
    image = Image.open(request.files.get("file"))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model_cardam.predict(data)
    return str('Car is Damaged!' if prediction[0][0]>.5 else 'Car is Good!')
if __name__=='__main__':
    app.run()
