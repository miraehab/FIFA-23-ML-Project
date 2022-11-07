import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pkl"), "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    positions = ['CF', 'CM', 'RW', 'GK', 'CB', 'LW', 'LM', 'LB','RM', 'RB']
    float_features = []

    #mms = MinMaxScaler()

    i = 0
    name = ""

    for x in request.form.values():
        if i == 0: 
            i += 1
            name = x
            continue
        float_features.append(float(x))
        """ if i == 0: 
            i += 1
            continue
        if type(x) != type("str"):
            
        else:
            if x in preferred_foot:
                encoded_data, mapping_index = pd.Series(preferred_foot).factorize()
                float_features.append(float(mapping_index.get_loc(x)))
            elif x in AttackingWorkRate:
                encoded_data, mapping_index = pd.Series(AttackingWorkRate).factorize()
                float_features.append(float(mapping_index.get_loc(x)))
            else:
                float_features.append(float(-1)) """

    features = [np.array(float_features)]
    #features = mms.fit_transform(features)
    prediction = model.predict(features)
    return render_template("result.html", prediction_text = "The Best Position for {} is {}".format(name, positions[prediction[0]]))

if __name__ == "__main__":
    flask_app.run(debug=True)