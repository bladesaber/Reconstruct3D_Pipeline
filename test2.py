import os
import shutil
import cv2
import matplotlib.pyplot as plt

src_dir = '/home/quan/Desktop/company/dataset/shape/raw'
mask_dir = '/home/quan/Desktop/company/dataset/shape/mask'
img_dir = '/home/quan/Desktop/company/dataset/shape/image'

non_process = []
# non_process = ['26.jpg', '11.jpg','27.jpg','23.jpg', '29.jpg', '38.jpg', '32.jpg', '4.jpg', '39.jpg']
# non_process = ['39.jpg']

if len(non_process)==0:
    file_names = os.listdir(src_dir)
else:
    file_names = non_process

for file_name in file_names:
    print(file_name)
    src_file = os.path.join(src_dir, file_name)
    save_file_name = file_name
    img_file = os.path.join(img_dir, save_file_name)
    mask_file = os.path.join(mask_dir, save_file_name)

    img = cv2.imread(src_file)

    # img = img[201:409, 0:210, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _, mask = cv2.threshold(gray, 180, maxval=255, type=cv2.THRESH_BINARY)

    # cv2.imwrite(img_file, img)
    # cv2.imwrite(mask_file, mask)

    filter_img = img.copy()
    filter_img[mask!=255, :] = 0

    cv2.imshow('rgb', img)
    cv2.imshow('gray', gray)
    cv2.imshow('filter', filter_img)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(0)
    if key==ord('s'):
        # cv2.imwrite(img_file, img)
        cv2.imwrite(img_file, filter_img)
        cv2.imwrite(mask_file, mask)
    elif key == ord('p'):
        non_process.append(file_name)
    elif key == ord('q'):
        break

print(non_process)
