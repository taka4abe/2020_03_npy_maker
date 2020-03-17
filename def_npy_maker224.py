#このコードがうまく走ることは確認したものの、データ量が24倍程度に爆発するので、
#このコードで学習データをまとめる方が良いと思われる。
#具体的には、178MB のデータが 6GB 程度になる。


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

def npy_maker(target_dir): # target_dirにstrを入れてnpyを返す。
    save_list_case = []
    for ID in IDnumber:
        ID = str(ID).zfill(3)
        save_list_ID = []
        for axes in four_axes:
            save_list_cycle = []
            for cycle in ten_cycle:
                cycle = str(cycle).zfill(2)
                file_name = target_dir + '/image_' + ID + '_' +\
                            axes + '_' + cycle + '.png'
                if not os.path.exists(file_name):
                    print(file_name, "not exists. Putting zeros")
                img = np.zeros((224, 224))
                try:
                    img = Image.open(file_name)
                    img = img.convert('L').resize((224, 224))
                    img = np.array(img) # dtype=uint8, shape:  (224, 224)
                except:
                    pass
                img = img.tolist()
                save_list_cycle.append(img)
            save_list_ID.append(save_list_cycle)
        save_list_case.append(save_list_ID)
    save_list_case = np.array(save_list_case) #(caseID, axis, cycle, (X, Y))
    saver = save_list_case.transpose(3, 4, 2, 0, 1) #((X, Y), cycle, caseID, axis)
    return saver

# making dataset
LAD  = npy_maker("pickup10_2018_LAD_120")
LCX  = npy_maker("pickup10_2018_LCX_120")
RCA  = npy_maker("pickup10_2018_RCA_120")
Nor  = npy_maker("pickup10_2018_N_120")
#LAD2 = npy_maker("pickup10_2018Data2_LAD")
#LCX2 = npy_maker("pickup10_2018Data2_LCX")
#RCA2 = npy_maker("pickup10_2018Data2_RCA")
#Nor2 = npy_maker("pickup10_2018Data2_N")
