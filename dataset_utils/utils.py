import os
import shutil

def recopy_img(src_dir, save_dir):
    paths = os.listdir(src_dir)
    for idx, path in enumerate(paths):
        file = os.path.join(src_dir, path)
        save_file = os.path.join(save_dir, '%d.jpg'%idx)

        shutil.copy(file, save_file)