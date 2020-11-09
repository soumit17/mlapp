import flask
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)

# main index page route
@app.route('/')
def home():
    return '<h1> API server is working! </h1>'

@app.route('/predict')
def predict():
    from sklearn.externals import joblib
    model=joblib.load('/home/ubuntu/mlapp/insurance.ml')
    insurance_cost=model.predict([[int(request.args['age']),
                                int(request.args['gender']),
                                int(request.args['bmi']),
                                int(request.args['children']),
                                int(request.args['smoker']),
                                int(request.args['region'])]])
    return str(round(insurance_cost[0],2))
if __name__=='__main__':
    app.run(debug=True)
