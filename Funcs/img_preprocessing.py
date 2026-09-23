import copy
import cv2
import numpy as np

def redink_remover(img):
    image = copy.deepcopy(img)
    # 获得红色通道
    blue_c, green_c, red_c = cv2.split(image)

    # 把图片转回 3 通道
    result_img = np.expand_dims( red_c, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

    return result_img


def draw_boxes(image, boxes):
    for box in boxes:
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image

def getSkewAngle(cvImage) -> float:
    h,w,_=cvImage.shape
    
    print('shape:',cvImage.shape)
    
    newImage = cvImage.copy()
    if min(h,w)>2000:
        scale=min(h,w)/1200
        print('scale:', )
        h=int(h/scale)
        w=int(w/scale)
        newImage=cv2.resize(newImage,(w,h),interpolation=cv2.INTER_LINEAR)
    img_=copy.deepcopy(newImage)
    # print('new shape:',newImage.shape)
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)

    x=min(h,w)//220
    if x%2==0:
        x=x-1
    print(x)
    blur = cv2.GaussianBlur(gray, (x,  x), 0)
#     _,thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x,  x))
    dilate = cv2.dilate(thresh, kernel, iterations=x)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    h_w=[]
    angs=[]
    if len(contours)>100:
        x=-80
    else:
        x=-5
    for c in contours[3:x]:
        minAreaRect= cv2.minAreaRect(c)
        ang=minAreaRect[-1]
        ww,hh=minAreaRect[1][0],minAreaRect[1][1]
        if abs(minAreaRect[1][0]-minAreaRect[1][1])<min(w,h)//30 or ww>w//1.3 or hh>h//1.3:
            continue
        
        points=cv2.boxPoints(minAreaRect)
#         points = sorted(list(cv2.boxPoints(minAreaRect)), key=lambda x: x[0])
        m=np.array(points, dtype='int32')
        if m[1][0]-m[0][0]<0 or m[3][1]-m[0][1]<0:#(m[3][1]-m[0][1])<0 or (m[1][0]-m[0][0])<0 or 
            print(m)

        a,b,c,d=int(m[0][0]) , int(m[0][1]), int(m[2][0]), int(m[2][1])

        if m[1][0]<=m[0][0]:
            m=[m[0],m[3],m[2],m[1]]
        if m[3][1]>=m[0][1]:
            m=[m[3],m[2],m[1],m[0]]

        h_w.append(abs(m[3][1]-m[0][1])-abs(m[1][0]-m[0][0]))
        img_=draw_boxes(img_,[m])
        angs.append(ang)
    h_w=sorted(h_w)[1:-1]
    if np.mean(h_w)>0:
        angle=90
    else:
        angle=0
    

    return  angle

def rotateImage(cvImage, angle: float):

    newImage = cvImage.copy()

    h,w = cvImage.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))

    M[0,2] += (rotated_w - w) // 2
    M[1,2] += (rotated_h - h) // 2


    newImage = cv2.warpAffine(newImage, M, (rotated_w,rotated_h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE) 
    return newImage
 
# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    print('angle:',angle)
    return rotateImage(cvImage, -1.0 * (angle) )



if __name__ == "__main__":
	img=cv2.imread(r'/data/tensorflow/kath/OCR/code/Temp/SMC自动化有限公司北京分公司.jpeg')
	res=deskew(img)
	print(res)



