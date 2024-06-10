from flask import Flask, request, jsonify
import os
import pandas as pd
from final_top import main  # Import the main function from your Python script
import base64
import final_top as scirpt_1
import final_spiral as script_2
import knn_ordering as script_3
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Get the selected option
    option = request.form['option']

    # Conditionally run different scripts based on the selected option
    if option == '1':
        output_data, images = scirpt_1.main(file_path)
    elif option == '2':
        output_data, images = script_2.main(file_path)
    elif option == '3':
        output_data, images = script_3.main(file_path)

    else:
        return jsonify({'error': 'Invalid option'})

    # Encode images as base64 strings
    encoded_images = {}
    for image_name, image_data in images.items():
        encoded_images[image_name] = base64.b64encode(image_data).decode('utf-8')

    # Return the processed data and encoded images as JSON response
    return jsonify({'data': output_data, 'images': encoded_images})

if __name__ == '__main__':
    app.run(debug=True)