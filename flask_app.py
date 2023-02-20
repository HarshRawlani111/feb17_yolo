from changing_dir import change_dir
from yolo_weights import download_weights
from repo_cloned import clone_repo
from predict import prediction
from flask import Flask, render_template, redirect, request, jsonify
import os
from flask_cors import CORS, cross_origin
from com_in_ineuron_ai_utils.utils import decodeImage

app = Flask(__name__)
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

CORS(app)

# @cross_origin()
class ClientApp:
    def __init__(self):
        self.read_image = "inputImage.jpg"
        # modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.predict = prediction(self.read_image)
        self.predict.preparing_inputs()
        self.predict.get_output_layers()
        self.predict.inference()
        self.predict.non_max_supperession()
        
        

@app.route("/")
def home():
    # return "Landing Page"
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()

def predictRoute():
    image = request.json['image']
    #change_dir().dir_yolo("/content/drive/MyDrive")
    #download_weights().weights()
    decodeImage(image, clApp.read_image)
    #clone_repo().clonning()
    #make sure all the files are current working directory
   
    clApp.predict.preparing_inputs()
    clApp.predict.get_output_layers()
    clApp.predict.inference()
    clApp.predict.non_max_supperession()

    


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    #app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=8000, debug=True)


