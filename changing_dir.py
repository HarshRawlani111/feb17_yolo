import os
from logging_app import yolo_logger
class change_dir:
  def __init__(self):
    self.log_writer = yolo_logger()
    self.filename = open("change_dir_logs.txt","a+")
    self.log_writer.log(self.filename, "dir changed {}".format(os.getcwd()))

  def dir_yolo(self, path):
    self.path = path
    new_location = os.makedir(path,"/yolo_folder")
    os.chdir(new_location)