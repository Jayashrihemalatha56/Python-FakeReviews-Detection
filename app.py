import os

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import random

app = Flask(__name__, static_folder='static')

CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')

    if not review:
        return jsonify({"error": "No review provided"}), 400

    # Fake review detection logic (for demonstration)
    prediction = "Fake" if len(review) % 2 == 0 else "Real"
    accuracy = round(random.uniform(85, 95), 2)  # Example accuracy range (85-95%)

    return jsonify({'result': prediction, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
