import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os 
import os.path
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout




app = Flask(__name__)
model = tf.keras.models.load_model("my_model")


@app.route('/') # Homepage
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    # data = request.get_json(force=True)
    # prediction = model.predict([[np.array(data['exp'])]])
    # output = prediction[0]
    # return jsonify(output)
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # making prediction
    return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction)) # rendering the predicted result

if __name__ == '__main__':
    app.run(debug=True)