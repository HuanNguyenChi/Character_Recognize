import cv2
import numpy as np
import tensorflow 
import pandas as pd

model = tensorflow.keras.models.load_model('E:\CODE\BTL\MyModel/Mymodel_With10Epochs2.h5')

word_dict= {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
            10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
            20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

def sol(path):
    img = cv2.imread(path , 1)
    img_copy = img.copy()

    img_copy = cv2.GaussianBlur(img_copy, (11,11), 0)
    # img_copy = cv2.resize(img_copy, (400, 400))
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    _,img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('image', img_thresh)
    # cv2.waitKey()

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, ( 28, 28)).astype('float32') / 255.0 
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    pre = model.predict(img_final)

    # in ra tỉ lệ 
    for i in range(26):    
        print(i, '%f'%(pre[0][i]*100), end=' ')
        print(word_dict[i])

    ans = word_dict[np.argmax(pre)]
    img = cv2.resize(img, (400,400))
    cv2.putText(img, "Prediction: " + ans, (40, 350), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 0, 0))
    cv2.imshow('image', img)
    cv2.waitKey()
    return word_dict[np.argmax(pre)]

# print(sol('E:\CODE\BTL\imgTest/a1.jpg'))