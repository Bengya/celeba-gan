import os
import cv2
import pickle
import numpy as np
import config as c

DATADIR = 'D:/Datasets/celeba-dataset/img_align_celeba/img_align_celeba/'
PICKLEFILE = 'real_imgs.pck'

def imgsToPickle():
    real_imgs = []

    img_folder= os.listdir(DATADIR)
    for img in img_folder[:100000]:
        path = os.path.join(DATADIR,img)
        img_array = cv2.imread(path, {3:cv2.IMREAD_COLOR,1:cv2.IMREAD_GRAYSCALE}[c.CHANNEL])
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        try:
            new_array = cv2.resize(img_array,(c.IMG_SIZE,c.IMG_SIZE))
            real_imgs.append(new_array)
        except:
            pass
    X = np.array(real_imgs).reshape(-1, c.IMG_SIZE, c.IMG_SIZE, c.CHANNEL)
    with open(PICKLEFILE,'wb') as file:
        pickle.dump(X, file)

def loadData():
    with open(PICKLEFILE,'rb') as file:
        return pickle.load(file)
        
if __name__ == '__main__':
    imgsToPickle()