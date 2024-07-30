from flask import render_template, request, redirect, url_for, jsonify
from app import app
from app.utils import allowed_file, predict_food
import os

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        predicted_class = predict_food(filepath)
        return jsonify({'predicted_class': predicted_class}), 200

    return jsonify({'error': 'Invalid file'}), 400
