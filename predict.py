import cv2
import argparse
import numpy as np
#from google.colab.patches import cv2_imshow #for colab
from logging_app import yolo_logger

class prediction:
  def __init__(self, read_image):
    self.log_writer = yolo_logger()
    self.logfile = open("PredictionLogs.txt","a+")
    self.log_writer.log(self.logfile, "Starting Prediction")
    self.read_image= read_image
    self.classes_args = "yolov3.txt"
    self.weights = "yolo3.weights"
    self.config = "yolov3.cfg"
    self.class_ids = []
    self.confidences = []
    self.boxes = []
    self.conf_threshold = 0.5
    self.nms_threshold = 0.4
    #we can either use above self or we can use args
  '''
  ap = argparse.ArgumentParser()
  ap.add_argument('-i', '--image', required=True,
                  help = 'path to input image')
  ap.add_argument('-c', '--config', required=True,
                  help = 'path to yolo config file')
  ap.add_argument('-w', '--weights', required=True,
                  help = 'path to yolo pre-trained weights')
  ap.add_argument('-cl', '--classes', required=True,
                  help = 'path to text file containing class names')
  args = ap.parse_args()
  '''
  #and in inputs we use "self" which is the variable-attachement that crosses over from one fun to another in same class
  # to call a fun we either call it like - function() or function(inputs) - if it takes inputs
  #all the variables that are common accross functions should be mentioned in "__init__" with self attahced so that we can call it in different functions    
  def preparing_inputs(self):
    try:
      '''
      self.read_image = read_image #Test image
      self.classes = classes 
      self.weights = weights #weights from pretrained model
      self.config = config
      '''
      
      self.log_writer.log(self.logfile, "loading image")
      # read input image
      self.image = cv2.imread(self.read_image)
      #Maybe this is needed to divide the image in equal blocks
      self.Width = self.image.shape[1] 
      self.Height = self.image.shape[0]
      scale = 0.00392 #(1/255 to scale the pixel values to [0..1])

      # read class names from text file
      self.log_writer.log(self.logfile, "reading classes")
      self.classes = None 
      with open(self.classes_args, 'r') as f: #opens this file name class
          self.classes = [line.strip() for line in f.readlines()]

      # generate different colors for different classes
      self.log_writer.log(self.logfile, "generating classes") 
      self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3)) # 3 colors generated randomly between 0 to 255

      # read pre-trained model and config file
      self.log_writer.log(self.logfile, "read pre-trained model and config file")
      self.net = cv2.dnn.readNet(self.weights, self.config)
      #dnn = deep learning neural network used only for predictions

      # create input blob 
      self.log_writer.log(self.logfile, "create input blob")
      blob = cv2.dnn.blobFromImage(self.image, scale, (416,416), (0,0,0), True, crop=False)
      self.log_writer.log(self.logfile, "created  blob")
      
      #416x416 square image, 
      # the mean value (default=0), 
      # the option swapBR=True (since OpenCV uses BGR to be converted to RGB)

      # set input blob for the network
      self.log_writer.log(self.logfile, "set input blob for the network")
      self.net.setInput(blob)
     
    
      # function to get the output layer names 
      # in the architecture
    except Exception as e:
      self.log_writer.log(self.logfile,"Error Occured {}".format(e))
      #self.log_writer.log(self.filename,e)

    



  def get_output_layers (self):
    
    try:

      self.log_writer.log(self.logfile, "get the output layer names ")
      layer_names = self.net.getLayerNames()
      '''
      try:
      
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
      except:
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
      '''
      self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    except Exception as e:
      self.log_writer.log(self.logfile,"Error Occured {}".format(e))
      #self.log_writer.log(self.filename,e)

    return self.output_layers

  def draw_prediction(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h, color, classes):
    self.image = img
    self.class_id = class_id
    self.confidence = confidence
    self.x = x
    self.y = y
    self.x_plus_w = x_plus_w
    self.y_plus_h = y_plus_h
    self.COLORS = color
    self.classes = classes
    #what ever variables that we are taking from other funs have to be mentioned in inputs 
    try:  
      self.log_writer.log(self.logfile, "draw bounding box on the detected object with class name ")
      pred = prediction(self.read_image,self.classes_args,self.weights,self.config)
      label = str(self.classes[self.class_id]) #Label on the bounding box is the class id

      color = self.COLORS[self.class_id]

      cv2.rectangle(self.image, (self.x,self.y), (self.x_plus_w,self.y_plus_h), color, 2)

      cv2.putText(self.image, label, (self.x-10,self.y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
      self.log_writer.log(self.logfile,"Error Occured {}".format(e))
      #self.log_writer.log(self.filename,e)
  
  def inference(self):
    try:
    # run inference through the network
    # and gather predictions from output layers
      self.log_writer.log(self.logfile, "running inference through the network and and gathering predictions from output layers ")
      outs = self.net.forward(self.output_layers)
      
      #outs = exactly where feed forward through the network happens ie 1D layer

      # initialization
      self.class_ids = []
      self.confidences = []
      self.boxes = []
      self.conf_threshold = 0.5
      self.nms_threshold = 0.4

      # for each detetion from each output layer 
      # get the confidence, class id, bounding box params
      # and ignore weak detections (confidence < 0.5)
      for out in outs:
          for detection in out:
              scores = detection[5:]
              self.class_id = np.argmax(scores)
              
              self.confidence = scores[self.class_id]
              if self.confidence > 0.5:
                  center_x = int(detection[0] * self.Width)
                  center_y = int(detection[1] * self.Height)
                  self.w1 = int(detection[2] * self.Width)
                  self.h1 = int(detection[3] * self.Height)
                  self.x1 = int(center_x - self.w1 / 2)
                  self.y1 = int(center_y - self.h1 / 2)
                  self.class_ids.append(self.class_id)
                  self.confidences.append(float(self.confidence))
                  self.boxes.append([self.x1, self.y1, self.w1, self.h1])
                  self.log_writer.log(self.logfile,"boxes list filled")
    except Exception as e:
      self.log_writer.log(self.logfile,"Error Occured {}".format(e))
      #self.log_writer.log(self.filename,e)



  
    


  def non_max_supperession(self):

    try:
      self.log_writer.log(self.logfile, "apply non-max suppression")
          # apply non-max suppression
      indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.conf_threshold, self.nms_threshold)
      self.log_writer.log(self.logfile, "running on each index")
      # go through the detections remaining
      # after nms and draw bounding box
      for i in indices:
                
        try:
          box = self.boxes[i]
        except:
          i = i[0]
        box =self.boxes[i]
        self.x = box[0]
        self.y = box[1]
        self.w = box[2]
        self.h = box[3]
        self.log_writer.log(self.logfile, "running on {} index,".format(i))
        predi = prediction(self.read_image,self.classes_args,self.weights,self.config)
        predi.draw_prediction(self.image, self.class_ids[i], self.confidences[i], round(self.x), round(self.y), round(self.x+self.w), round(self.y+self.h), self.COLORS, self.classes)
        #unlike calling a variable from a function (which is done by self), if we want to call a function we do it by "class.function" and all the variables that would be used in it have to be mentioned explicitely as inputs since it wont take those values directly from other function, but if we specify those values in inputs even if used with self (like above) for 'self.COLORS, self.classes' 
        #we mention "class.function" to call that function 
      self.log_writer.log(self.logfile, "object-detection.jpg created")
      ''' 
      # display output image    
      #cv2.imshow("object detection", preparing_inputs.image)
      #for colab
      cv2_imshow(self.image)
      # wait until any key is pressed
      cv2.waitKey()
          
      # save output image to disk
      cv2.imwrite("object-detection.jpg", self.image)
      self.log_writer.log(self.filename, "saved output image to disk")
      # release resources
      cv2.destroyAllWindows()
      '''
    except Exception as e:
        self.log_writer.log(self.logfile,"Error Occured {}".format(e))
        #self.log_writer.log(self.filename,e)
    return cv2.imwrite("object-detection.jpg", self.image)
    
    