from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)
model = tf.keras.models.load_model("my_model")

@app.route('/') # Homepage
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1,-1)
    print(final_features)
    prediction = model.predict(final_features) # making prediction
    return render_template('index.html', prediction_text='Percent chance that this transaction is fraud: {}'.format(prediction)) # rendering the predicted result

if __name__ == '__main__':
    app.run(debug=True)