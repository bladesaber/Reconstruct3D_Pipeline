import os
import shutil

dir = '/home/quan/Desktop/company/temp/20220712_091454'

fake2_dir = '/home/quan/Desktop/company/temp/data/fake2'
fake4_dir = '/home/quan/Desktop/company/temp/data/fake4'
real_dir = '/home/quan/Desktop/company/temp/data/real'

# for idx in os.listdir(dir):
#     idx_dir = os.path.join(dir, idx)
#
#     for name in os.listdir(idx_dir):
#         if 'cut_2' in name:
#             new_name = 'fake_%s.jpg'%idx
#             new_path = os.path.join(fake2_dir, new_name)
#             from_path = os.path.join(idx_dir, name)
#             shutil.copy(from_path, new_path)
#
#         elif 'cut_4' in name:
#             new_name = 'fake_%s.jpg' % idx
#             new_path = os.path.join(fake4_dir, new_name)
#             from_path = os.path.join(idx_dir, name)
#             shutil.copy(from_path, new_path)
#
#         elif 'orig' in name:
#             new_name = 'real_%s.jpg' % idx
#             new_path = os.path.join(real_dir, new_name)
#             from_path = os.path.join(idx_dir, name)
#             shutil.copy(from_path, new_path)