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
type_list = list(anime['type'].values)
output_path = 'archive/cv_staff'
exist_paths = glob(output_path + '/*')
exist_ids = []
for ep in exist_paths:
    if '\\' in ep:
        ep = ep.replace('\\', '/')
    exist_ids.append(int(ep.split('/')[-1].split('_')[0]))
exist_ids = set(exist_ids)

# anime_names = list(anime['name']) # name有问题不能用


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


def extract_cv(cv_data):
    cv_data = [r for r in cv_data.text.split('\n') if r]
    cv_data_list = []
    for i in range(len(cv_data) // 4):
        # 有时候某些角色CV会没有列出，后续的都不会列出，所以如果得不到国籍，
        # 或者应该出现国籍的地方出现Main,Supporting等词提前跳出
        if i * 4 + 3 >= len(cv_data) or cv_data[i * 4 + 3] in ('Main', 'Supporting') \
                or cv_data[i * 4 + 1] not in ('Main', 'Supporting'):
            break
        cv_data_list.append({'chara_name': cv_data[i * 4],
                             'chara_type': cv_data[i * 4 + 1],
                             'cv_name': cv_data[i * 4 + 2],
                             'country': cv_data[i * 4 + 3]})
    return cv_data_list


def extract_staff(staff_data):
    staff_data = [r for r in staff_data.text.split('\n') if r]
    staff_data_list = []
    for i in range(len(staff_data) // 2):
        staff_data_list.append({'staff_name': staff_data[i * 2],
                                'staff_type': staff_data[i * 2 + 1]})
    return staff_data_list


for i_, anime_id in enumerate(tqdm(anime_ids)):
    if type_list[i_] not in {'TV', 'OVA', 'Movie'}:
        continue
    if anime_id in exist_ids:
        print('Exist:{}'.format(anime_id))
        continue
    try:
        anime = jikan.anime(anime_id)
        name = anime['url'].split('/')[-1]
        url = anime['url']
        print('ID:{}, Name:{}'.format(anime_id, name))
        total_reviews = []
        try:
            html = getHTMLText(url)
            soup = BeautifulSoup(html, 'html.parser')
            cv_staff_data = soup.find_all(attrs={'class': 'detail-characters-list clearfix'})
            if len(cv_staff_data) >= 1:
                cv_list = extract_cv(cv_staff_data[0])
            else:
                cv_list = []
            if len(cv_staff_data) == 2:
                staff_list = extract_staff(cv_staff_data[1])
            else:
                staff_list = []

            print('CV:{} STAFF:{}'.format(len(cv_list), len(staff_list)))

            data = {'cv': cv_list, 'staff': staff_list}
            with open(output_path + '/{}_{}.json'.format(anime_id, name), 'w', encoding='utf-8') as w:
                json.dump(data, w, ensure_ascii=False, indent=2)

            time.sleep(1)

        except Exception as e:
            error_num += 1
            print('ERROR:', e)
            print('Can not get cv_staff from {}:{}'.format(anime_id, name))
            error_sleep(error_num)

    except Exception as e:
        error_num += 1
        print('ERROR:', e)
        print('Can not get name from {}'.format(anime_id))
        error_sleep(error_num)
