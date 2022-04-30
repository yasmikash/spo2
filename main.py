# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('models/finalized_model.pkl','rb'))
@app.route('/oxygenlevel',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([np.array(data['data'])])
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)