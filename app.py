import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('behaviour_pickle.pkl','rb'))


@app.route('/')
def startup():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html',prediction_text= 'Driver Behaviour currently : {}'.format(output))

if __name__=='__main__':
    app.run(debug = True)
