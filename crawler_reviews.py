import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from tqdm import tqdm
from glob import glob
from jikanpy import Jikan

jikan = Jikan()


def getHTMLText(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""


anime = pd.read_csv('./archive/anime.csv')
anime_ids = list(anime['anime_id'].values)
output_path = 'archive/reviews'
exist_paths = glob(output_path + '/*')
exist_ids = []
for ep in exist_paths:
    if '\\' in ep:
        ep = ep.replace('\\', '/')
    exist_ids.append(int(ep.split('/')[-1].split('_')[0]))
exist_ids = set(exist_ids)


# anime_names = list(anime['name']) # name有问题不能用

def Filter(reviews):
    rev_res = []
    for r in reviews:
        lines = r.text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        # filter words
        filter_words = ('Overall', 'Story', 'Animation', 'Sound', 'Character', 'Enjoyment')
        res = ""
        for line in lines:
            if not line.isdigit() and line not in filter_words:
                res += (" " + line)
        rev_res.append(res)

    return rev_res


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
        name = anime['url'].split('/')[-1]
        print('ID:{}, Name:{}'.format(anime_id, name))
        total_reviews = []
        try:
            for i in range(1, 50):
                html = getHTMLText('https://myanimelist.net/anime/{}/{}/reviews?p={}'.format(anime_id, name, i))
                soup = BeautifulSoup(html, 'html.parser')
                reviews = soup.find_all(attrs={'class': 'spaceit textReadability word-break pt8 mt8'})
                reviews = Filter(reviews)
                print('Page:{} reviews:{}'.format(i, len(total_reviews) + len(reviews)))
                if len(reviews) == 0:
                    break
                else:
                    total_reviews.extend(reviews)
                if len(reviews) < 20:
                    break

            data = {'anime_id': int(anime_id), 'name': name, 'reviews': total_reviews}
            with open(output_path + '/{}_{}.json'.format(anime_id, name), 'w', encoding='utf-8') as w:
                json.dump(data, w, ensure_ascii=False, indent=2)

            print('Complete {}:{} with {} reviews'.format(anime_id, name, len(total_reviews)))
            time.sleep(1.2)

        except Exception as e:
            error_num += 1
            print('ERROR:', e)
            print('Can not get reviews from {}:{}'.format(anime_id, name))
            error_sleep(error_num)

    except Exception as e:
        error_num += 1
        print('ERROR:', e)
        print('Can not get name from {}'.format(anime_id))
        error_sleep(error_num)
