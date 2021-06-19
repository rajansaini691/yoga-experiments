
"""
Downloads part of the dataset (we'll do 8 poses)

If we need more training examples, just remove this logic: 'if i > 30'
"""
import os
import wget
import urllib
import shutil
import PIL
from PIL import Image

# Stores filename; chosen at random
pose_names = [
    'Akarna_Dhanurasana',
    'Boat_Pose_or_Paripurna_Navasana_',
    'Bound_Angle_Pose_or_Baddha_Konasana_',
    'Bow_Pose_or_Dhanurasana_',
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana_',
    'Camel_Pose_or_Ustrasana_',
    'Cat_Cow_Pose_or_Marjaryasana_',
    'Chair_Pose_or_Utkatasana_']

os.makedirs('./data', exist_ok=True)

num_training = 40
num_test = 10

shutil.rmtree('./data/')
os.makedirs('./data/')

for pose in pose_names:
    with open('./yoga_dataset_links/' + pose + '.txt') as f:
        i = 0
        train_or_test = "train"
        for line in f.readlines():
            if i > num_training:
                train_or_test = "test"
            if i > num_training + num_test:
                break

            path, image_url = line.split('\t')

            folder = path.split('/')[0]
            os.makedirs(f"./data/{train_or_test}/{folder}", exist_ok=True)

            try:
                wget.download(image_url, out=f"./data/{train_or_test}/{path}")

                # Make sure the file is openable by PIL
                with Image.open(f"./data/{train_or_test}/{path}") as img:
                    pass

                i += 1
                print(i)

            except PIL.UnidentifiedImageError:
                os.remove(f'./data/{train_or_test}/{path}')
            except:
                pass

        print('Finished pose ' + pose)
