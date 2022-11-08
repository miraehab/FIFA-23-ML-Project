import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pkl"), "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    positions = ['CF', 'CM', 'RW', 'GK', 'CB', 'LW', 'LM', 'LB','RM', 'RB']
    preferred_foot = ['Left', 'Right']
    AttackingWorkRate = ['High', 'Low', 'Medium']
    float_features = []
    minmax = np.array([[16, 44],[155, 206],[54, 105],[47, 91],[48, 95],
    [0, 26], [759, 2305],[224, 502],[0, 190500000],[500, 450000],
    [0, 366700000],[2022.0, 2032.0],[False, True],[0, 1],[1, 5],
    [1, 5],[1, 5],[0, 2],[0, 2],[28, 97],[16, 92],[25, 93],[28, 93],
    [15, 91],[30, 91],[6, 92],[3, 94],[5, 91],[10, 93],[3, 90],[3, 95],
    [6, 93],[4, 94],[10, 93],[5, 94],[14, 97],[15, 97],[18, 94],[30, 94],
    [20, 95],[18, 94],[27, 95],[15, 95],[25, 96],[4, 90],[10, 95],[3, 90],
    [2, 95],[10, 92],[6, 92],[13, 95],[4, 92],[7, 92],[6, 90],[2, 90],[2, 90],
    [2, 93],[2, 91],[2, 90]])

    i = 0
    name = ""

    for x in request.form.values():
        if i == 0: 
            i += 1
            name = x
            continue
        if x in preferred_foot:
            encoded_data, mapping_index = pd.Series(preferred_foot).factorize()
            float_features.append(float(mapping_index.get_loc(x)))
        elif x in AttackingWorkRate:
            encoded_data, mapping_index = pd.Series(AttackingWorkRate).factorize()
            float_features.append(float(mapping_index.get_loc(x)))
        else:
            float_features.append(float(x))

    features = [np.array(float_features)]
    print(features)
    normalize_dataset(features, minmax)
    #features = mms.fit_transform(features)
    prediction = model.predict(features)
    return render_template("result.html", prediction_text = "The Best Position for {} is {}".format(name, positions[prediction[0]]))

if __name__ == "__main__":
    flask_app.run(debug=True)