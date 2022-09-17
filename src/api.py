from flask import Flask, jsonify
from flask import request
from flask import send_file
import cv2
import numpy as np
import json
import remove_pd_module

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Hello, World!</h1>"


@app.route('/addFace', methods = ['POST'])
def add_face():
   if request.method == 'POST':
      image_file = request.files['image'].read()
      ret = remove_pd_module.add_face(image_file)
      if ret == None:
        return jsonify({})

      data = {'uuid': ret}

      return jsonify(data)

@app.route('/getIgnoreFaceFeature', methods=['GET'])
def get_ignore_face_feature():
    if request.method == 'GET':
        uuid = request.form['uuid']
        print(uuid)
        ret = remove_pd_module.get_ignore_face_feature(uuid)
        if len(ret) == 0:
            print('Nonedddddd')
            return jsonify({})
        data = {'feature': ret}
        return jsonify(data)

@app.route('/getBlurredImage', methods = ['POST'])
def get_blurred_image():
   if request.method == 'POST':
      image_file = request.files['image'].read()
      # original_image = Image.open(file.file)
      nparr = np.fromstring(image_file, np.uint8)
      cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      cv_img.save('test.jpeg','jpeg')



      return 'file:///C:/Users/user/Desktop/final/remove_pd/src/test.jpeg'

@app.route('/getBlurredImage2', methods = ['POST'])
def get_blurred_image2():
   if request.method == 'POST':
      image_file = request.files['image'].read()
      uuids = request.form['uuid']

      uuidList=[]
      
      ret = remove_pd_module.get_blurred_image2(image_file, uuids)
      if ret == None:
        return jsonify({"no"})

      data = {'uuid': ret}

      return jsonify(data)

#test ocr
@app.route('/getOcrImage', methods = ['POST'])
def get_ocr_image():
   if request.method == 'POST':
      image_file = request.files['image'].read()


      uuidList=[]
      
      ret = remove_pd_module.get_ocr_image(image_file)
      if ret == None:
        return jsonify({"no"})

      data = {'uuid': ret}

      return ret   

@app.route('/Img', methods = ['POST'])
def img():
   
      image_file = request.files['image'].read()
      img = cv2.imread('C:\\Users\\user\\Desktop\\final\\mp\\acc1.jpg', cv2.IMREAD_COLOR)
      
      source = r"C:\\Users\\user\\Desktop\\final\\mp\\acc.jpg"
      
      
      nparr = np.fromstring(image_file, np.uint8)
      img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #   img = process_img(img) 
    
      return json.dumps(img.tolist())
# @app.route('/addFace', methods = ['POST'])
# def add_face():
#    if request.method == 'POST':
#       f = request.files['file']
#       ret = remove_pd_module.add_face(f)

#       return {''}

# @app.route('/addFace', methods = ['POST'])
# def add_face():
#    if request.method == 'POST':
#       f = request.files['file']
#       ret = remove_pd_module.add_face(f)

#       return {''}


# @app.route('/addFace', methods = ['POST'])
# def add_face():
#    if request.method == 'POST':
#       f = request.files['file']
#       ret = remove_pd_module.add_face(f)

#       return {''}

@app.route('/user/<name>')
def user(name):
	return '<h1>Hello, {0}!</h1>'.format(name)

if __name__ == '__main__':
    app.run(port=5001, debug=True)