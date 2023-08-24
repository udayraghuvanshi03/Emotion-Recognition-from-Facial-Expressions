import numpy as np
import pandas as pd
import os
from PIL import Image

df = pd.read_csv('C:/Users/udayr/PycharmProjects/MLfiles/Project_data/train.csv')
df_test=pd.read_csv('C:/Users/udayr/PycharmProjects/MLfiles/Project_data/test.csv')
df_all=pd.read_csv('C:/Users/udayr/PycharmProjects/MLfiles/Project_data/icml_face_data.csv')
df0 = df[df['emotion'] == 0]
df1 = df[df['emotion'] == 1]
df2 = df[df['emotion'] == 2]
df3 = df[df['emotion'] == 3]
df4 = df[df['emotion'] == 4]
df5 = df[df['emotion'] == 5]
df6 = df[df['emotion'] == 6]

os.mkdir("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Angry/")
os.mkdir("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Disgust/")
os.mkdir("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Fear/")
os.mkdir("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Happy/")
os.mkdir("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Sad/")
os.mkdir("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Surprise/")
os.mkdir("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Neutral/")


For all images in one folder
d=0
for image_pixels in df.iloc[1:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/All images/img_%d.jpg"%d, "JPEG")
    d+=1


For all test images
print(len(df_test.iloc[0:,0]))
d=0
for image_pixels in df_test.iloc[0:,0]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/All_test_images/img_%d.jpg"%d, "JPEG")
    d+=1

# For all train images
image_pixels=df.iloc[0,1].split(' ')
print(len(image_pixels))
d=0
for image_pixels in df.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/All_train_images/img_%d.jpg"%d, "JPEG")
    d+=1
d=0
for image_pixels in df0.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Angry/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df1.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Disgust/img_%d.jpg"%d, "JPEG")
    d+=1


d=0
for image_pixels in df2.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Fear/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df3.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Happy/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df4.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Sad/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df5.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Surprise/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df6.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Neutral/img_%d.jpg"%d, "JPEG")
    d+=1

d=0
for image_pixels in df7.iloc[0:,1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    img = Image.fromarray(image_data)
    img.save("C:/Users/udayr/PycharmProjects/MLfiles/Project_data/Angry_happy_sad/img_%d.jpg"%d, "JPEG")
    d+=1
