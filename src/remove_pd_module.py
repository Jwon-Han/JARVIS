import io
from sys import flags
import uuid
from typing import List

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
from PIL import ImageFont, ImageDraw, Image
from PIL import ImageFilter
import easyocr
# # import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh

# from insightface.data import get_image as ins_get_image
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
face_threshold = 0.6

ignore_face_list = dict()


def init_ignore_face_list():

    pass

def get_face_feature(cv_img):
    # nparr = np.fromstring(img, np.uint8)
    # cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # pil_img = Image.open(img)
    # np_img=np.array(pil_img) 
    # cv_img=cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


    faces = app.get(cv_img)
    print('================={}'.format(len(faces)))

    if len(faces) != 1:
        print('face not found')
        return []

    return faces[0].normed_embedding

def get_valid_faces(cv_img, ignoreFaceList: List[List[float]]):
    # nparr = np.fromstring(img, np.uint8)
    # cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = app.get(cv_img)
    if len(faces) == 0:
        return []

    feats = []
    for face in faces:
        feats.append(face.normed_embedding)
    feats = np.array(feats, dtype=np.float32)
    
    # sims = np.dot(feats, feats.T)
    ignoreIdxList = np.ones((len(faces),), dtype=int)
    for ignoreFace in ignoreFaceList:
        nd_ig_face = np.array(ignoreFace)
        sims = np.dot(nd_ig_face, feats.T)
        print(sims)
        ignores = np.nonzero(sims > face_threshold)

        for ignore in ignores:
            ignoreIdxList[ignore] = 0

    # get final faces
    final_faces = []
    for idx, ignoreIdx in enumerate(ignoreIdxList):
        if ignoreIdx == 1:
            final_faces.append(faces[idx])

    return final_faces


def add_face(cv_img, uuid):
    
    face_feature = get_face_feature(cv_img)
    if len(face_feature) == 0:
        return None

    # face_feature_id = str(uuid.uuid4())
    face_feature_id = uuid

    # store (uuid, face feature) to memory and DB

    ignore_face_list[face_feature_id] = face_feature.tolist()
    return face_feature_id

def get_ignore_face_list():
    return ignore_face_list

def get_ignore_face_feature(uuid: str):

    if uuid in ignore_face_list.keys():
        return ignore_face_list[uuid]

    return []


def remove_face(uuid: str):
    pass


def update_face(uuid: str, img : io.BytesIO):

    face_feature = get_face_feature(img)


    pass


def get_blurred_image3(img, key, ignoreFaceUUIDList: List[str]):
    ignoreFaceFeatureList = []
    for ignoreFaceUUID in ignoreFaceUUIDList:
        feature = get_ignore_face_feature(ignoreFaceUUID)
        if len(feature) == 0:
            continue

        ignoreFaceFeatureList.append(feature)

    feats=[]
    faces = get_valid_faces(img, ignoreFaceFeatureList)
    print("---- valid face {}".format(len(faces)))
    
    flm=[]
    for face in faces:
        facelandmarks = []
        # face_landmark.append(face.landmark_2d_106)
        # print(face.landmark_2d_106)
        for face_landmarks in face.landmark_2d_106:
            
            # for i in range(face.landmark_2d_106):
                # pt1 = face_landmarks[i]
                # print(face_landmarks[0])
                x = int(face_landmarks[0])
                y = int(face_landmarks[1])
                facelandmarks.append([x, y])
        flm.append(facelandmarks)

    height, width, _ = img.shape
    result = img.copy()
    for face in range(len(faces)):
        frame_copy = result.copy()

        # 1. Face landmarks detection
        landmarks = np.array(flm[face], np.int32)
        convexhull = cv2.convexHull(landmarks)

        # 2. Face blurrying
        mask = np.zeros((height, width), np.uint8)
        # cv2.polylines(mask, [convexhull], True, 255, 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        # Extract the face
        frame_copy = cv2.blur(frame_copy, (27, 27))
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)


        # Extract background
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(result, result, mask=background_mask)

        # Final result
        result = cv2.add(background, face_extracted)


    result_path = "C:/Users/user/Desktop/final/remove_pd/pic/blur/blur"+key+".jpeg"
    cv2.imwrite(result_path, result)

    return result_path

def get_ocr_image(img, s, removeTexts: List[str]=None ):

    cv2.imwrite("C:/Users/user/Desktop/final/remove_pd/pic/ocr"+s+".jpeg", img)
    def ocr(i):
        img = cv2.imread("C:/Users/user/Desktop/final/remove_pd/pic/ocr"+s+".jpeg", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # Create rectangle mask
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        

        x = result[i][0][0][0]
        y = result[i][0][0][1]
        w = result[i][0][1][0] - result[i][0][0][0]
        h = result[i][0][2][1] - result[i][0][1][1]

        draw.rectangle(((x, y), (x+w, y+h)), fill=255, width=2)

        blurred = img.filter(ImageFilter.GaussianBlur(52))

        # Paste blurred region and save result
        img.paste(blurred, mask=mask)
        images = np.array(img)

        images = cv2.cvtColor(images,cv2.COLOR_BGR2RGB)
        cv2.imwrite("C:/Users/user/Desktop/final/remove_pd/pic/ocr"+s+".jpeg", images)

    reader = easyocr.Reader(['ko', 'en'])
    result =  reader.readtext("C:/Users/user/Desktop/final/remove_pd/pic/ocr"+s+".jpeg")
    words=[]

    for i in range(len(result)):
        words.append(result[i][1].replace(" ",""))
    print(words)
    for i in range(len(words)):
        if '356-0770-3851-13' in words[i]:
            ocr(i)
        if '전지수' in words[i]:
            ocr(i)
    # cv2.imread("C:/Users/user/Desktop/final/remove_pd/pic/ocr"+s+".jpeg", cv2.IMREAD_COLOR)
    return "C:/Users/user/Desktop/final/remove_pd/pic/ocr"+s+".jpeg"

def img(img):
    return img


init_ignore_face_list()