from flask import Flask, request, jsonify
import os
import base64
from final_top import main  # Import the main function from your Python script
import final_top as script_1
import final_spiral as script_2
import knn_ordering as script_3
import limit_knn as script_4
import new_ordering as script_5
import individual as script_6
import new as script_7
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

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
        output_data, images = script_1.main(file_path)
    elif option == '2':
        output_data, images = script_2.main(file_path)
    elif option == '3':
        output_data, image_data_list = script_3.main(file_path)
    elif option == '4':
        output_data, images = script_4.main(file_path)
    elif option == '5':
        output_data, images = script_5.main(file_path)
    elif option == '6':
        output_data, image_data_list = script_6.main(file_path)
    elif option == '7':
        output_data, images = script_7.main(file_path)
    else:
        return jsonify({'error': 'Invalid option'})

    # Encode images as base64 strings
    # code for encoding images from list of images
    if option == '3' or option == '6':
        print("option 3 or 6")
        encoded_images = {}
        for i, image_data in enumerate(image_data_list):
            encoded_images[f'image_{i}'] = base64.b64encode(image_data).decode('utf-8')
    else:
        encoded_images = {}
        # print number of items in image
        for image_name, image_data in images.items():
            encoded_images[image_name] = base64.b64encode(image_data).decode('utf-8')

    # Return the processed data and encoded images as JSON response
    return jsonify({'data': output_data, 'images': encoded_images})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
