import os
import pickle
from src.utils import *

save_dir = 'data/preprocessed'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# zipping images into dictionaries
print('zipping images for 2021...')
dict_h_21 = zip_images('data/hyper/2021', panels=False, div_100=True)
dict_m_21 = zip_images('data/multi/2021', panels=False)
dict_chm_21 = zip_images('data/structure/chm/2021', panels=False)
dict_chm_lidar_21 = zip_images('data/structure/chm_lidar/2021', panels=False)

print('zipping images for 2022...')
dict_h_22 = zip_images22('data/hyper/2022', div_100=True)
dict_m_22 = zip_images22('data/multi/2022')
dict_m_22['20220810'][30][0,45,62] = np.nan    # removing the error pixel in the image
dict_chm_22 = zip_images22('data/structure/chm/2022')
dict_chm_lidar_22 = zip_images22('data/structure/chm_lidar/2022')

dict_h = {
    '2021': dict_h_21,
    '2022': dict_h_22
}
dict_m = {
    '2021': dict_m_21,
    '2022': dict_m_22
}
dict_chm = {
    '2021': dict_chm_21,
    '2022': dict_chm_22,

    '2021_lidar': dict_chm_lidar_21,
    '2022_lidar': dict_chm_lidar_22
}
# saving the dictionaries
pickle.dump(dict_h, open(os.path.join(save_dir,'zipped_h.pkl'), 'wb'))
pickle.dump(dict_m, open(os.path.join(save_dir,'zipped_m.pkl'), 'wb'))
pickle.dump(dict_chm, open(os.path.join(save_dir,'zipped_chm.pkl'), 'wb'))
print('zipping images completed!!! Files saved in ' + save_dir)