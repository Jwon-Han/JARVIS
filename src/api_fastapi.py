from importlib.metadata import requires
from typing import Union
import os
from PIL import Image
from io import BytesIO
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import json
from fastapi.encoders import jsonable_encoder
import remove_pd_module
from PIL import Image, ImageFilter
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Response,Form
import io
import base64
from starlette.responses import StreamingResponse
app = FastAPI()
from fastapi.responses import FileResponse
path = "C:\\Users\\user\\Desktop\\final\\mp"
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post('/addFace')
async def add_face(file: UploadFile, uuid: str = Form()):
    original_image = Image.open(file.file)
    np_img = np.array(original_image)
    cv_img=cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    ret = remove_pd_module.add_face(cv_img, uuid)
    if ret == None:
        return jsonable_encoder({'result':False})

    data = {'result': True}
    return jsonable_encoder(data)

@app.route('/getIgnoreFaceFeature', methods=['GET'])
def get_ignore_face_feature(uuid: str = Form()):
    # if request.method == 'GET':
        # uuid = request.form['uuid']
        print(uuid)
        ret = remove_pd_module.get_ignore_face_feature(uuid)
        if len(ret) == 0:
            print('Nonedddddd')
            return jsonable_encoder({})
        data = {'feature': ret}
        return jsonable_encoder(data)

@app.get('/images/thumbnail/{filename}',
  response_description="Returns a thumbnail image from a larger image",
  response_class="StreamingResponse",
  responses= {200: {"description": "an image", "content": {"image/jpeg": {}}}})
def thumbnail_image (filename: str):
  # read the high-res image file
  image = Image.open(filename)
  # create a thumbnail image
  image.thumbnail((100, 100))
  imgio = io.BytesIO()
  image.save(imgio, 'JPEG')
  imgio.seek(0)
  return StreamingResponse(content=imgio, media_type="image/jpeg")

@app.post('/Img')
async def local(file: UploadFile):
    
    original_image = Image.open(file.file)
    s=datetime.now().strftime('%y%m%d%H%M%S')
    original_image.save("C:\\Users\\user\\Desktop\\final\\remove_pd\\pic\\"+s+'.jpeg','jpeg')
    path = os.path.abspath(s+'.jpeg')
    prettypath="C:/Users/user/Desktop/final/remove_pd/pic/"+s+".jpeg"

    return prettypath

@app.post("/login/")
async def login(file: UploadFile):
    
    original_image = Image.open(file.file)
    

    filtered_image = BytesIO()
    original_image.save(filtered_image, "JPEG")
    filtered_image.seek(0)

    return StreamingResponse(filtered_image, media_type="image/jpeg")

@app.post('/getOcrImage')
# def get_blurred_image():
async def get_ocr_image(file: UploadFile):    
#    if request.method == 'POST':
    original_image = Image.open(file.file)
    s=datetime.now().strftime('%y%m%d%H%M%S')
    original_image.save("C:\\Users\\user\\Desktop\\final\\remove_pd\\pic\\"+s+'.jpeg','jpeg')
    path = os.path.abspath(s+'.jpeg')
    prettypath="C:/Users/user/Desktop/final/remove_pd/pic/"+s+".jpeg"
    img = cv2.imread(prettypath, cv2.IMREAD_COLOR)
    ret = remove_pd_module.get_ocr_image(img,s)

    return ret


@app.post('/getBlurredImage')
# def get_blurred_image():
async def get_blurred_image(file: UploadFile, uuid: str = Form()):    
#    if request.method == 'POST':
    # img2 = file2
    original_image = Image.open(file.file)
    np_img = np.array(original_image)
    cv_img=cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    key=datetime.now().strftime('%y%m%d%H%M%S')
    original_image.save("C:\\Users\\user\\Desktop\\final\\remove_pd\\pic\\"+key+'.jpeg','jpeg')
    #prettypath="C:/Users/user/Desktop/final/remove_pd/pic/"+s+".jpeg"
    #img = cv2.imread(prettypath, cv2.IMREAD_COLOR)

    

    uuids = uuid.split(',')
    ret = remove_pd_module.get_blurred_image3(cv_img, key, uuids)

    return ret
# img = file.file.read()
    # img=Image.open(file.file)
    # nparr = np.fromstring(img, np.uint8)
    # cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # return_img = cv2.processImage(cv_img)
    # _, encoded_img = cv2.imencode('.PNG', return_img)
    # encoded_img = base64.b64encode(encoded_img)

    # return img


    # original_image = original_image.filter(ImageFilter.BLUR)