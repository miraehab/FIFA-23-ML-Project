import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("Deployment\model4.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    positions = ['CF', 'CM', 'RW', 'GK', 'CB', 'LW', 'LM', 'LB','RM', 'RB']
    """preferred_foot = ['Left', 'Right']
    AttackingWorkRate = ['High', 'Low', 'Medium'] """
    float_features = []

    #mms = MinMaxScaler()

    i = 0

    for x in request.form.values():
        if i == 0: 
            i += 1
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
    return render_template("index.html", prediction_text = "The Best Position for this Player is {}".format(positions[prediction[0]]))

if __name__ == "__main__":
    flask_app.run(debug=True)