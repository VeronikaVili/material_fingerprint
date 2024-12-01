# Flask web server libraries
from flask import Flask, render_template, request
# Import system library - path creation
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import time
from source import compute_all, prepare_image_photo

# path - path to image location
def open_vid_image(path):
    # Reading image
    image = cv2.imread(path)
    # Converting BGR input to RGB image
    image = image[...,::-1]
    return image

def save_vid_image(path, image):
    image = image[..., ::-1]
    cv2.imwrite(path, image)

app = Flask(__name__)

img_folder = os.path.join('static', 'images')
app.config['UPLOAD'] = img_folder

@app.route('/', methods=['GET', 'POST'])
def index():  # put application's code here
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/display', methods=['GET', 'POST'])
def display():
    if request.method == 'POST':
        file = request.files["img1"]
        img1 = get_img(file)
        file = request.files["img2"]
        img2 = get_img(file)
        return render_template('display.html', image1=img1, image2=img2)
    return render_template('display.html')
def get_img(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD'], filename))
    img = os.path.join(app.config['UPLOAD'], filename)
    return img

def crop_image(data, image):
    t = int(data['top'])
    l = int(data['left'])
    w = int(data['width'])
    h = int(data['height'])
    return image[t:(t + h), l:(l + w), :]
@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        img = open_vid_image(data['crops'][0]['image'])
        img = crop_image(data['crops'][0], img)
        img = prepare_image_photo(img)
        save_vid_image((os.path.join(app.config['UPLOAD'], 'cropped_image_1.png')), img)

        img = open_vid_image(data['crops'][1]['image'])
        img = crop_image(data['crops'][1], img)
        img = prepare_image_photo(img)
        save_vid_image((os.path.join(app.config['UPLOAD'], 'cropped_image_2.png')), img)
        return render_template('index.html')
    return render_template('index.html')

@app.route('/check', methods=['GET', 'POST'])
def check():
    img1 = (os.path.join(app.config['UPLOAD'], 'cropped_image_1.png'))
    img2 = (os.path.join(app.config['UPLOAD'], 'cropped_image_2.png'))
    return render_template('check.html', image1=img1, image2=img2)

@app.route('/compute', methods=['GET', 'POST'])
def compute():
    # indicates the time delay caused due to processing
    img1 = (os.path.join(app.config['UPLOAD'], 'cropped_image_1.png'))
    img2 = (os.path.join(app.config['UPLOAD'], 'cropped_image_2.png'))
    compute_all(img1, img2, os.path.join(app.config['UPLOAD'], 'polar_plot.png'))
    return render_template('result.html', image1=os.path.join(app.config['UPLOAD'], 'polar_plot.png'))

if __name__ == '__main__':
    app.run()
