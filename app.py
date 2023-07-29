from flask import Flask, render_template, request
import os
import tensorflow as tf
from prediction import PredictionPipeline
from PIL import Image
import numpy as np
from utils import get_cropped_image_if_2_eyes
import cv2


app = Flask(__name__)

UPLOAD_FOLDER = 'testing_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method=='GET':
        return render_template('index.html')
    else:

        if 'image' not in request.files:
            return "No file part in the request."
        
        image_file = request.files['image']
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename))
        print(image_file.filename)
        try:

            cropped_image = get_cropped_image_if_2_eyes(os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename))
            print(cropped_image)
            cropped_file_name = "cropped_img"+".png"
            cropped_file_path = UPLOAD_FOLDER + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, cropped_image) 
            

            img= Image.open(cropped_file_path)
            print(img.size)
            print(np.array(img).shape)
            new_img=img.resize((256,256))
            new_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'modified.jpg'))
            print(new_img.size)
            print(np.array(new_img).shape)
            np_img = np.array(new_img)
        

            pred_obj = PredictionPipeline(np_img)
            predicted_class, confidence = pred_obj.predict()
            


            for f in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, f))


            return render_template('index.html',predicted_class=predicted_class, confidence=confidence)

        except Exception as e:
            ex_msg = "Please provide a clear image"
            return render_template('index.html',ex_msg=ex_msg)

if __name__ == '__main__':
    app.run(debug=True)