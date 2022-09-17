import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import remove_pd_module

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)


# then print all-to-all face similarity
feats = []
for face in faces:
    feats.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)

for ig_face in feats[:2]:
    sims = np.dot(ig_face, feats.T)
    print(sims)
    print(np.nonzero(sims >0.1))
    print('')


remove_pd_module.get_blurred_image("peel.jpg", "f244068d-8359-4633-988a-e9b3750a3cbb")