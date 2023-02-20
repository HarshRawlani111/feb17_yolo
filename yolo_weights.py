import requests
from logging_app import yolo_logger

class download_weights:
  def weights(self):
    
    self.log_writer = yolo_logger()
    self.filename = open("weightsLogs.txt","a+")
    self.log_writer.log(self.filename, "downloading yolov3 weights")
    try:
        weights_url = 'https://pjreddie.com/media/files/yolov3.weights'   
        r = requests.get(weights_url)
        with open("yolo3.weights",'wb') as f:
            f.write(r.content)
        self.log_writer.log(self.filename, " yolov3 weights dowloaded")
    except Exception as e:
      self.log_writer.log(self.filename, e)