from aocr.util import dataset
dataset.generate_from_custom('datasets/annotations.txt', 'datasets')

'''import cv2
img_path = 'D:/Development/Big_Data_and_Machine_Learning/ANPR/licenseplate_labeler/dataset/images/20200713-123327.png'
with open(img_path, 'rb') as img_file:
    img = img_file.read()
    print(img)
    print(type(img))

img2 = cv2.imread(img_path)
b_img = bytes(img2)
print(b_img)
print(type(b_img))'''