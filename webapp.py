import os
import shutil
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import io
import preprocess.prepare_image as preprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'myImages'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if os.name == 'nt':
    os.system("rmdir /s /q dataset")  # Command to remove a directory on Windows
else:
    os.system("rm -rf dataset")  # Command to remove a directory on Unix-like systems

# Create the dataset directories
os.mkdir("dataset")
os.mkdir("dataset/data")
os.mkdir("dataset/data/test_single")
os.mkdir("dataset/landmark")
os.mkdir("dataset/landmark/ALL")
os.mkdir("dataset/mask")
os.mkdir("dataset/mask/ALL")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return 'No selected file', 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)

    # Process the image (example: convert it to grayscale)
    preprocess.prepare_image(file_path)

    return 'Image processed successfully'

@app.route('/run_test_script', methods=['GET'])
def run_test_script():
    os.system("python test.py --dataroot dataset/data/test_single --name formal_author --model test --dataset_mode single --norm batch --use_local --which_epoch 300 --gpu_ids -1")
    return 'Test script executed'

@app.route('/results/<path:filename>')
def serve_results(filename):
    print(filename)
    return send_from_directory('results/', filename)

if __name__ == '__main__':
    app.run(debug=True)
