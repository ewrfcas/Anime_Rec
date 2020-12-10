import pandas as pd
from tqdm import tqdm
from glob import glob
from jikanpy import Jikan
import os
import urllib
import time

jikan = Jikan()

anime = pd.read_csv('./archive/anime.csv')
anime_ids = list(anime['anime_id'].values)
output_path = r'archive/imgs'
os.makedirs(output_path, exist_ok=True)
exist_paths = glob(os.path.join(output_path, '*'))
exist_ids = []
for ep in exist_paths:
    if '\\' in ep:
        ep = ep.replace('\\', '/')
    exist_ids.append(int(ep.split('/')[-1].split('_')[0]))
exist_ids = set(exist_ids)

error_num = 0


def error_sleep(error_num):
    if error_num % 5 == 0:
        print('sleep for 10s')
        time.sleep(10)
    elif error_num % 17 == 0:
        print('sleep for 63s')
        time.sleep(63)
    else:
        time.sleep(1.3)


for anime_id in tqdm(anime_ids):
    if anime_id in exist_ids:
        print('Exist:{}'.format(anime_id))
        continue
    try:
        anime = jikan.anime(anime_id)
        img_url = anime['image_url']
        name = anime['url'].split('/')[-1]
        urllib.request.urlretrieve(img_url, filename=output_path + '/{}_{}.jpg'.format(anime_id, name))
        time.sleep(1.2)

    except Exception as e:
        error_num += 1
        print('ERROR:', e)
        print('Can not get img from {}'.format(anime_id))
        error_sleep(error_num)
