import cv2
import numpy as np
import matplotlib.pyplot as plt

orig = cv2.imread('/home/psdz/HDD/quan/output/test/result/20220707_154741/3/orig.jpg')
cut2 = cv2.imread('/home/psdz/HDD/quan/output/test/result/20220707_154741/3/cut_2.jpg')
cut4 = cv2.imread('/home/psdz/HDD/quan/output/test/result/20220707_154741/3/cut_4.jpg')
cut8 = cv2.imread('/home/psdz/HDD/quan/output/test/result/20220707_154741/3/cut_8.jpg')

orig_blur = cv2.GaussianBlur(orig, ksize=(3, 3), sigmaX=1, sigmaY=1)
cut2 = cv2.GaussianBlur(cut2, ksize=(3, 3), sigmaX=1, sigmaY=1)
cut4 = cv2.GaussianBlur(cut4, ksize=(3, 3), sigmaX=1, sigmaY=1)

orig = orig_blur

from models.RIAD.loss_utils import MSGMSLoss
import torch

msgm_loss_fn = MSGMSLoss(num_scales=4)

orig_d = (np.transpose(orig, (2, 0, 1))[np.newaxis, ...]).astype(np.float32) / 255.
cut2_d = (np.transpose(cut2, (2, 0, 1))[np.newaxis, ...]).astype(np.float32) / 255.
cut4_d = (np.transpose(cut4, (2, 0, 1))[np.newaxis, ...]).astype(np.float32) / 255.
# cut8_d = (np.transpose(cut8, (2, 0, 1))[np.newaxis, ...]).astype(np.float32) / 255.

orig_d = torch.from_numpy(orig_d)
cut2_d = torch.from_numpy(cut2_d)
cut4_d = torch.from_numpy(cut4_d)
# cut8_d = torch.from_numpy(cut8_d)

r_cut2 = msgm_loss_fn(orig_d, cut2_d, as_loss=False)
r_cut2 = r_cut2.cpu().numpy()
r_cut2 = np.transpose(r_cut2[0, ...], (1, 2, 0))
r_cut2 = r_cut2[:, :, 0]

r_cut4 = msgm_loss_fn(orig_d, cut4_d, as_loss=False)
r_cut4 = r_cut4.cpu().numpy()
r_cut4 = np.transpose(r_cut4[0, ...], (1, 2, 0))
r_cut4 = r_cut4[:, :, 0]

r = (r_cut2 + r_cut4)/2.0

r_max = r.max()
r = (r - r.min())/(r.max()-r.min())
# plt.figure('r')
# plt.imshow(r)
# plt.show()

defect_mask = np.zeros(r.shape, dtype=np.uint8)
defect_mask[r>0.3] = 255

show_img = orig.copy()
contours, hierarchy = cv2.findContours(defect_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c_id in range(len(contours)):
    cv2.drawContours(show_img, contours, c_id, (0, 0, 255), 1)
cv2.putText(show_img, 'Score:%f' % r_max, org=(5, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            color=(0, 0, 255), thickness=1)

cv2.imshow('orig', orig)
cv2.imshow('show', show_img)
cv2.imshow('cut2', cut2)
cv2.waitKey(0)