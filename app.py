# -*- coding: utf-8 -*-

import numpy as np

from flask import Flask,request,render_template,jsonify
import pickle
app=Flask(__name__)
model=pickle.load(open("model_ar.pkl","rb"))
@app.route("/")
def home():
    return render_template('home.html')
@app.route("/predict",methods=["POST"])
def predict():
    int_features=[float(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    output=round(prediction[0])
    return render_template("home.html",prediction_text="price of the edtech course should be {}".format(output))
if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)



