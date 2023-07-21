#!/usr/bin/env python
# coding: utf-8

# In[57]:


import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy import array
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder

import os


# In[27]:


img = cv2.imread('C:/Users/iljoogns/Documents/classification/OK1.jpg')
img = cv2.imread('C:/Users/iljoogns/Documents/classification/OK1.jpg',1) # 톤 변경
img = cv2.imread('C:/Users/iljoogns/Documents/classification/OK1.jpg',2)


# In[28]:


type(img)


# In[29]:


img


# In[31]:


img.shape


# In[6]:


from matplotlib import pyplot as plt


# * cv2.IMREAD_COLOR : 컬러 이미지를 로드, 이미지의 투명성 무시, 기본 플래그
# * cv2.IMREAD_GRAYSCALE : 이미지를 회색조 모드로 로드
# * cv2.IMREAD_UNCHANGED : 알파 채널을 포함하여 이미지를 로드

# In[30]:


plt.imshow(img)


# ### 이미지 저장

# In[ ]:


cv2.imwrite('/', img)


# ### 이미지 변경

# In[ ]:


re_image1 = cv2.resize(img, (84,84), interpolation=fv2.INTER_AREA) # 색 중간 톤을 어느정도 할지 정하는 것 ; interploation


# ### 이미지 자르기

# In[35]:


# croppedImage = img.crop(1850, 390, 2345, 700 )

croppedImage2 = img[450:750, 1850:2350]

plt.imshow(croppedImage2)


# ### 이미지 자른 후, train 폴더/OK 폴더에 저장

# In[49]:


for i in range(1, 11):
    dirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/origin/train/OK/'
    imageName = 'OK'+ str(i)+'.jpg'
    
    print(dirPath +imageName)
    img = cv2.imread(dirPath + imageName)
    
    croppedImage = img[450:750, 1850:2350]
    
    
#     plt.imshow(croppedImage)

    cv2.imwrite('C:/Users/iljoogns/Documents/dayouep/genesis/cropped/train/'+ imageName, croppedImage)


# In[73]:


def saveImage(dirPath, imageName, newDirPath, start, end):
    
    for i in range(start, end +1):
        
        fileName = imageName + str(i) + '.jpg'
        img = cv2.imread(dirPath + fileName)
        
        croppedImage = img[450:750, 1850:2350]
        
#         print(newDirPath + fileName)
        
        cv2.imwrite(newDirPath + fileName, croppedImage)


# ### 양품 train 데이터 저장

# In[3]:


dirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/origin/train/OK/'
imageName = 'OK'

newDirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/cropped/train/OK/'

saveImage(dirPath, imageName, newDirPath, 1, 36000)


# In[92]:


fig = plt.figure()
rows = 1
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
img = cv2.imread('C:/Users/iljoogns/Documents/dayouep/genesis/cropped/train/OK/OK1.jpg')
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Origin')
ax1.axis("off")

print(img.shape)

ax2 = fig.add_subplot(rows, cols, 2)
resizeImg = cv2.resize(img, (250,250), interpolation=cv2.INTER_AREA) # 색 중간 톤을 어느정도 할지 정하는 것 ; interploation
ax2.imshow(cv2.cvtColor(resizeImg, cv2.COLOR_BGR2RGB))
ax2.set_title('Resize')
ax2.axis("off")

print(resizeImg.shape)

plt.show()


# ### 양품 test 데이터 저장

# In[95]:


dirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/origin/test/OK/'
imageName = 'OK'

newDirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/cropped/test/OK/'

saveImage(dirPath, imageName, newDirPath, 2601, 3348)


# ### 불량 train 데이터 저장

# In[97]:


dirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/origin/train/NG/'
imageName = 'NG'

newDirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/cropped/train/NG/'

saveImage(dirPath, imageName, newDirPath, 1, 2600)


# ### 불량 test  데이터 저장

# In[98]:


dirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/origin/test/NG/'
imageName = 'NG'

newDirPath = 'C:/Users/iljoogns/Documents/dayouep/genesis/cropped/test/NG/'

saveImage(dirPath, imageName, newDirPath, 2601, 3348)


# ### 학습 데이터 변환

# In[4]:


trainDir = 'C:/Users/iljoogns/Documents/dayouep/genesis/cropped/train/'


# In[5]:


trainFolderList = array(os.listdir(trainDir))


# In[42]:


trainInput = []
trainLabel = []


# In[43]:


labelEncoder = LabelEncoder()
integerEncoded = labelEncoder.fit_transform(trainFolderList) # 폴더를 라벨 1,2로 구분

oneHotEncoder = OneHotEncoder(sparse = False) # False로 하면 y가 OK 1 NG 0 형식으로 나온다 True는 0,1,0,1 식으로 출력
integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)
oneHotEncoded = oneHotEncoder.fit_transform(integerEncoded)


# In[44]:


print(integerEncoded)
print(oneHotEncoded)


# In[45]:


for i in range(len(trainFolderList)):
    path = os.path.join(trainDir, trainFolderList[i])
    path = path + '/'
    imgList = os.listdir(path)
    
    for img in imgList:
        imgPath = os.path.join(path, img)
        print(imgPath)
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    #         img = cv2.imread(imgPath)
        trainInput.append([np.array(img)])
        trainLabel.append([np.array(oneHotEncoded[i])])


# In[46]:


# print(trainInput)
len(trainInput)
# print(trainLabel)


# ### CNN 데이터 형태로 저장

# In[55]:


cTrainInput = np.reshape(trainInput, (-1, 64, 64))
cTrainInput = np.array(cTrainInput).astype(np.float32)

trainLabel = np.reshape(trainLabel, (-1, 2))
trainLabel = np.array(trainLabel).astype(np.float32)
np.save("cTrainInput.npy", cTrainInput)


# In[48]:


print(cTrainInput.shape)
print(cTrainInput[0]) # input 형식  : (데이터 행 갯수, 픽셀 shape, 채널 수)


# ### 테스터 데이터 변환

# In[63]:


testDir = 'C:/Users/iljoogns/Documents/dayouep/genesis/cropped/test/'
testFolderList = array(os.listdir(trainDir))

testInput = []
testLabel = []

labelEncoder = LabelEncoder()
integerEncoded = labelEncoder.fit_transform(testFolderList) # 폴더를 라벨 1,2로 구분

oneHotEncoder = OneHotEncoder(sparse = False) # False로 하면 y가 OK 1 NG 0 형식으로 나온다 True는 0,1,0,1 식으로 출력
integerEncoded = integerEncoded.reshape(len(integerEncoded), 1)
oneHotEncoded = oneHotEncoder.fit_transform(integerEncoded)

for i in range(len(trainFolderList)):
    path = os.path.join(testDir, testFolderList[i])
    path = path + '/'
    imgList = os.listdir(path)
    
    for img in imgList:
        imgPath = os.path.join(path, img)
        print(imgPath)
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    #         img = cv2.imread(imgPath)
        testInput.append([np.array(img)])
        testLabel.append([np.array(oneHotEncoded[i])])
        
cTestInput = np.reshape(testInput, (-1, 64, 64))
cTestInput = np.array(cTestInput).astype(np.float32)
np.save("cTrainInput.npy", cTestInput)


# ### 모델 생성

# In[50]:


print(cTrainInput.shape)
print(cTestInput.shape)

cTrainInput = cTrainInput.reshape(-1,64,64,1)
cTestInput = cTestInput.reshape(-1,64,64,1)


# In[51]:


print(cTrainInput.shape, cTestInput.shape)


# In[52]:


cnnModel = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(64,64,1), kernel_size=(3,3), filters = 16), # kernel
    # stride : 1 이 기본 값
    tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 32),
    tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 64),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 128, activation = 'relu'),
    tf.keras.layers.Dense(units = 2, activation = 'softmax')
])

cnnModel.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001),
                loss='categorical_crossentropy', metrics=['accuracy'])

cnnModel.summary()


# ### 모델 훈련

# In[56]:


cnnHistory = cnnModel.fit(cTrainInput, trainLabel, epochs = 25)


# In[60]:


plt.figure(figsize = (12,4))

plt.subplot(1,2,1)
plt.plot(cnnHistory.history['loss'], 'b-', label='loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(cnnHistory.history['accuracy'], 'g-' , label='accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()


# ### CNN 모델 검증

# In[64]:


cnnModel.evaluate(cTestInput, testLabel)


# In[ ]:


cPredTest = cnnModel.predict_classes(cTestInput)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusionTest = confusionMatrix(newTestLabel, cPredTest)

