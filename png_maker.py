import os
from PIL import Image
import numpy as np

total = 0
for folder_name in os.listdir('.'):
    if os.path.isdir(folder_name) == True:
        total += 1
    else:
        pass
print("total:", total, "\n")


counter = 0
for folder_name in os.listdir('.'):
    if os.path.isdir(folder_name) == True:
        counter += 1
        print("Now working on", folder_name, counter, '/', total)
        for file_name in os.listdir(folder_name):
            rawfile = folder_name + '/' + file_name
            if os.path.isdir(rawfile) == False:

                root, ext =     os.path.splitext(file_name)
                save_name =     folder_name + '/' + root + '.png'

                tmp =           open(rawfile, 'rb')
                img_data =      tmp.read()
                tmp.close()

                Imagesize =     (120, 120)

                img =           Image.frombytes('F', Imagesize, img_data)
                npimg =         np.array(img)
                npimg_255 =     npimg / np.max(npimg) * 255
                npimg_255 =     Image.fromarray(npimg_255.astype(np.float32)
                                               ).convert('RGB')
                npimg_255.save(save_name)
            else:
                pass
    else:
        pass

print("\nFinish converting raw to png.")
print("\n'rm */*.raw' on bash, is recommended, thank you.\n\nFin.")
