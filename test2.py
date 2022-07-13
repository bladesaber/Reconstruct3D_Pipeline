import os
import shutil
import numpy as np

dir = '/home/psdz/HDD/quan/output/test/result/20220713_131439'
real_dir = '/home/psdz/HDD/quan/temp_trash/fakeVSreal/real'
fake_dir = '/home/psdz/HDD/quan/temp_trash/fakeVSreal/fake'

for idx in os.listdir(dir):
    idx_dir = os.path.join(dir, idx)

    for name in os.listdir(idx_dir):
        if 'cut_2' in name:
            from_path = os.path.join(idx_dir, name)
            to_path = os.path.join(fake_dir, 'fake%s.jpg'%idx)
            shutil.copy(from_path, to_path)
        elif 'orig' in name:
            from_path = os.path.join(idx_dir, name)
            to_path = os.path.join(real_dir, 'real%s.jpg'%idx)
            shutil.copy(from_path, to_path)

