from git import Repo
from logging_app import yolo_logger
import os

class clone_repo:
  def clonning(self):
    
    self.log_writer = yolo_logger()
    self.filename = open("RepoClonnedLogs.txt","a+")
    self.log_writer.log(self.filename, "Starting to clone Repo")

    try:
        git_url = "https://github.com/arunponnusamy/object-detection-opencv.git"
        dir = os.getcwd()
        Repo.clone_from(git_url, dir)
        self.log_writer.log(self.filename, "Repo clonned")
        self.filename.close()
    except Exception as e:
      self.log_writer.log(self.filename, "unable to clone repo: {}".format(e))
      self.filename.close()