from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/food_classification_model.h5')
classes = ['apple_pie', 'baby_back_ribs', '...']  # Add all your class names here

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            img = load_img(filepath, target_size=(150, 150))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            prediction = model.predict(x)
            predicted_class = classes[np.argmax(prediction)]
            return f'Predicted class: {predicted_class}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
