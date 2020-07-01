import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    result = str();
    if output==1:
        result = 'Positive'
    elif output==0:
        result = 'Negative'
    else:
        output = 'Unpredicted value'
    return render_template('index.html', prediction_text='Test result is {}'.format(result))


if __name__ == "__main__":
    app.run(debug=True)