from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)


tomato_classes= ['Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf Spot', 'Tomato Spider Mites Two Spotted Spider Mite', 'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato Mosaic Virus', 'Tomato Healthy']

def get_result(file):
    tomatoModel=load_model("finalTomato.h5")

    img=image.load_img(file, target_size=(256,256))
    process = image.img_to_array(img)
    process = np.expand_dims(process, axis=0)
    res = tomatoModel.predict(process)
    return tomato_classes[np.argmax(res)]



@app.route('/')
def base():
    print("1st Page.")
    return render_template("base.html")


@app.route('/output', methods=['GET', 'POST'])
def output():
    print("2nd Page.")
    f = request.files['inputImage']
    print(type(f))

    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir,'static', 'uploads', secure_filename(f.filename))
    f.save(file_path)

    output = get_result(file_path)
    output=str(output)
    file_name = os.path.basename(file_path)
    tempres={"opt":output,"file":file_name}
    return render_template("result.html", result=tempres)


app.run(debug=False)

