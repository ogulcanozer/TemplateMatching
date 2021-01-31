import cv2 as cv
import numpy as np
import math
import sys
from matplotlib import pyplot as plt


def draw_angled_rec(verts, img):
    for i in range(len(verts)-1):
        cv.line(img, (verts[i][0], verts[i][1]), (verts[i+1][0],verts[i+1][1]), (255,255,255), 2)
    cv.line(img, (verts[3][0], verts[3][1]), (verts[0][0], verts[0][1]), (255,255,255), 2)

def rotate_bound(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv.warpAffine(image, M, (nW, nH)), (nW,nH)

#PROGRAM START
if len(sys.argv) < 2:
    print(f"No file. Usage :{sys.argv[0]} <StarMap> <Template>")
    sys.exit(1)

img = cv.imread(sys.argv[1],0)
template = cv.imread(sys.argv[2],0)
w= template.shape[0]
h= template.shape[1]

best_max_val = 0
best_max_loc = None
best_angle = 0

for angle in np.arange(0, 360, 5):
    rotated, newLengths = rotate_bound(template, angle)
    res = cv.matchTemplate(img,rotated,cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if max_val>best_max_val:
        best_max_val = max_val
        best_max_loc = max_loc
        best_angle = angle
        centerX = best_max_loc[0] + (newLengths[0]//2)
        centerY = best_max_loc[1] + (newLengths[1]//2)
        M = cv.getRotationMatrix2D((centerX, centerY), angle, 1.0)
        orgRec = np.array([[centerX - w//2, centerY - h//2, 1],[centerX + w//2, centerY - h//2,1 ],[centerX + w//2, centerY + h//2,1 ],[centerX - w//2, centerY + h//2,1]],np.int32)
        nep = np.dot(M,orgRec.T)



print(nep)


