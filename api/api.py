#! /bin/env python

import time
import base64
from io import BytesIO

from PIL import Image
from flask import Flask, request
from infere import model, infere

app = Flask(__name__)

def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    img_base64 = "data:image/png;base64," + img_str
    return img_base64


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method != 'POST':
        return {}
    img = request.files['file']
    #io.BytesIO
    img = Image.open(img)
    img = img.convert('RGB')
    print(img)
    #img = encode_image(img)

    labels, images, descriptions = infere(img, model)
    images = [encode_image(img) for img in images]

    return {'labels': labels, 'descriptions': descriptions, 'images': images}

if __name__ == '__main__':
    app.run()
