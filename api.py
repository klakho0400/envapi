import requests
from flask import Flask
import jsonify
import json
import os
from pymongo import MongoClient
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



from boto3.session import Session
import boto3

ACCESS_KEY = 'AKIAYI2Q75UE6EJNHGSG'
SECRET_KEY = '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'

session = Session(aws_access_key_id=ACCESS_KEY,
              aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')
your_bucket = s3.Bucket('environmentdetection')

for s3_file in your_bucket.objects.all():
    print(s3_file.key) # prints the contents of bucket

s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY , aws_secret_access_key=SECRET_KEY)
s3.download_file('environmentdetection','WhatsApp Video 2020-06-23 at 10.16.48 AM.3gpp','out.mp4')

model = load_model('narmodel.h5')
source=cv2.VideoCapture('out.mp4')
indoor=0
outdoor=0
while(True):
    ret,img=source.read()
    if not ret:
        break
    resized=cv2.resize(img,(128,128))  #re-sizing the image
    new_img=preprocess_input(resized)
    reshaped=np.reshape(new_img,(1,128,128,3))
    result=model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    if label==0:
        indoor+=1
    else:
        outdoor+=1
cv2.destroyAllWindows()
source.release()

#code to convert video to audio till sound export


print(os.getcwd())
# v = os.open('out.mp4',os.O_RDONLY)

app=Flask(__name__)
text = []
cluster = MongoClient("mongodb+srv://chatteltech19:chattel19@cluster0-icted.mongodb.net/machine_model?retryWrites=true&w=majority")
db = cluster["machine_model"]
collection = db["inputs"]

"""To display input text in json format in a route"""

"""Main machine learning function calling here and retrieving text from database"""


@app.route('/api', methods=['GET','POST'])
def environment_detect():
    if indoor>outdoor:
        return 0
    else:
        return 1
   

print(environment_detect())