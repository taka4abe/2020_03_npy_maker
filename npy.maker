# (120, 120)...resolution
# (0-10)...cycle
# (001..051)... case ID
# (0, 1, 2, 3)...short, 2, 3, 4 chamber view
import os
import numpy as np
from PIL import Image

# example file_name: LAD/image_050_s_09.png
dir_names = ('pickup10_2018Data2_LAD',  'pickup10_2018_LAD_120',
             'pickup10_2018Data2_LCX',  'pickup10_2018_LCX_120',
             'pickup10_2018Data2_N',    'pickup10_2018_N_120',
             'pickup10_2018Data2_RCA',  'pickup10_2018_RCA_120')
IDnumber = list(range(1, 51, 1))
four_axes = ('s', '2', '3', '4')
ten_cycle = list(range(10))

for dir in dir_names:
    save_list_case = []
    for ID in IDnumber:
        ID = str(ID).zfill(3)
        save_list_ID = []
        for axes in four_axes:
            save_list_cycle = []
            for cycle in ten_cycle:
                cycle = str(cycle).zfill(2)
                file_name = dir + '/image_' + ID + '_' +\
                            axes + '_' + cycle + '.png'
                #if os.path.exists(file_name):
                img = np.zeros(120**2).reshape(120, 120) # ファイルがない場合にzerosで埋める
                try:
                    img = Image.open(file_name)
                    img = img.convert('L') # (120, 120) L (0, 255)
                    img = np.array(img) # dtype=uint8, shape:  (120, 120)
                except:
                    pass
                img = img.tolist()
                save_list_cycle.append(img)
            save_list_ID.append(save_list_cycle)
        save_list_case.append(save_list_ID)
    save_list_case = np.array(save_list_case)
    print(save_list_case.shape) # (50, 4, 10, 120, 120)
    saver = save_list_case.transpose(3, 4, 2, 0, 1) #((X, Y), cycle, caseID, axis)
    print(saver.shape) # (120, 120, 10, 4, 50)
    save_name = dir + '.npy'
    print(save_name)
    np.save(save_name, saver)
