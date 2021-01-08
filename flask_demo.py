from flask import Flask, request, jsonify
import numpy as np
import os
import json
import collections
import pickle
import pandas as pd

app = Flask(__name__, template_folder='flask_temp', static_folder='CF_data')

like_score = 0.7
hate_score = -0.5
topk = 20
return_num = 10
rating_to_tag = 0.4
rev_rep = 5  # review repeat times
main_score = 2.0
sup_score = 1.0
staff_score = 0.3
rating_to_cv = 0.2
piv_norm_path = os.path.join('CF_data', 'piv_norm.pkl')

print('Loading data...')
anime_info = pd.read_csv(os.path.join('archive', 'anime.csv'))
piv_norm = pickle.load(open(piv_norm_path, 'rb'))
print('Index mapping...')
piv_anime = pd.DataFrame(piv_norm.index)
piv_anime = piv_anime.merge(anime_info, left_on='name', right_on='name')
piv_map = {piv_anime['anime_id'][i]: {'name': piv_anime['name'][i],
                                      'genre': piv_anime['genre'][i],
                                      'type': piv_anime['type'][i],
                                      'rating': piv_anime['rating'][i],
                                      'piv_idx': i} for i in range(piv_anime.shape[0])}
valid_type = {'TV', 'OVA', 'Movie'}
print('Loading Tags...')
unique_tags = []
for tag in anime_info['genre'].values:
    if type(tag) != str:
        continue
    for t_ in tag.split(','):
        unique_tags.append(t_.strip())
unique_tags = np.unique(unique_tags)
print('Total tags', len(unique_tags))
print(unique_tags)
tag_map = {tag: i for i, tag in enumerate(unique_tags)}
tag_map_rev = {i: tag for i, tag in enumerate(unique_tags)}
tag_matrix = np.zeros((len(piv_map), len(unique_tags) + 1))  # 多出来的一维特征用于匹配score
for anime_id in piv_map:
    piv_idx = piv_map[anime_id]['piv_idx']
    tag_matrix[piv_idx, len(unique_tags)] = float(piv_map[anime_id]['rating'])  # 最后一位放score
    tags = piv_map[anime_id]['genre']
    if type(tags) == str:
        tags = tags.split(',')
        for tag in tags:
            tag_matrix[piv_idx, tag_map[tag.strip()]] = 1.0

# load reviews
print('Loading reviews...')
review_data = json.load(open(os.path.join('archive', 'review_data.json')))
review_word_map = json.load(open(os.path.join('archive', 'review_word_map.json')))
review_word_rev_map = {}
for w in review_word_map:
    review_word_rev_map[review_word_map[w]] = w
review_map = {}
review_rev_map = {}
review_mat = []
for i, name in enumerate(review_data):
    rid = int(name.split('_')[0])
    review_map[rid] = i
    review_rev_map[i] = name
    rm = np.zeros((1, len(review_word_map)))
    for w in review_data[name]:
        if w[0] in review_word_map:
            rm[0, review_word_map[w[0]]] = w[1]
    review_mat.append(rm)
review_mat = np.concatenate(review_mat, axis=0)

# load cv staff
print('Loading cv staffs...')
cv_staff_data = json.load(open(os.path.join('archive', 'cv_staff_data.json')))
cv_anime_map = {}
cv_anime_rev_map = {}
cvstaff_map = json.load(open(os.path.join('archive', 'cv_staff_mapping.json')))
cvstaff_rev_map = {}
cvstaff_mat = np.zeros((len(cv_staff_data), len(cvstaff_map)))
for name in cvstaff_map:
    cvstaff_rev_map[cvstaff_map[name]] = name
for i, name in enumerate(cv_staff_data):
    rid = int(name.split('/')[-1].split('_')[0])
    cv_anime_map[rid] = i
    cv_anime_rev_map[i] = name
    # 主角CV
    for cv_name in cv_staff_data[name]['main']:
        cvstaff_mat[i, cvstaff_map[cv_name]] = main_score
    # 配角CV
    for cv_name in cv_staff_data[name]['sup']:
        cvstaff_mat[i, cvstaff_map[cv_name]] = sup_score
    # staff
    for staff_name in cv_staff_data[name]['staff']:
        cvstaff_mat[i, cvstaff_map[staff_name]] = staff_score
print('Init over...')


@app.route("/get_rec", methods=['POST'])
def get_rec():
    # user_info:{id1:0,id2:1,id3:1}
    user_info = json.loads(request.form['user_info'])
    print('Recieve', user_info)
    user_vec = np.zeros([1, 5479])
    user_tags = np.zeros([1, len(unique_tags) + 1])
    user_reviews = np.zeros([1, len(review_word_map)])
    user_cvstaff = np.zeros([1, len(cvstaff_map)])
    for anime_id in user_info:
        anime_id_int = int(anime_id)
        if anime_id_int not in piv_map:
            print('动画ID:{}不在piv_map内'.format(anime_id))
            continue

        if user_info[anime_id] == 1:
            user_vec[0, piv_map[anime_id_int]['piv_idx']] = like_score
            # 喜欢的话，追加tag
            user_tags[0, :] += tag_matrix[piv_map[anime_id_int]['piv_idx'], :]
            if anime_id_int in review_map:
                user_reviews[0, :] += review_mat[review_map[anime_id_int], :]
            if anime_id_int in cv_anime_map:
                user_cvstaff[0, :] += cvstaff_mat[cv_anime_map[anime_id_int], :]
        elif user_info[anime_id] == 0:
            user_vec[0, piv_map[anime_id_int]['piv_idx']] = hate_score
        else:
            raise NotImplementedError
    # 归一化tag
    user_tags[0, -1] = 0
    user_tags_sorted = np.argsort(user_tags[0])[::-1]
    print('User tags:')
    for uts in user_tags_sorted:
        if user_tags[0, uts] > 0:
            print(tag_map_rev[uts], user_tags[0, uts])
    user_tags = user_tags / np.max(user_tags)
    user_tags[0, -1] = rating_to_tag
    # 输出用户review关键词
    user_reviews_sorted = np.argsort(user_reviews[0])[::-1]
    # review逐条过滤
    user_reviews = np.tile(user_reviews, [rev_rep, 1])
    print('User keywords in reviews:')
    for i, urs in enumerate(user_reviews_sorted[:20]):
        if i + 1 < rev_rep:
            user_reviews[i + 1:, urs] = 0
        if user_reviews[0, urs] > 0:
            print(review_word_rev_map[urs], user_reviews[0, urs])
    # 输出用户偏爱的CV和STAFF
    user_cvstaff_sorted = np.argsort(user_cvstaff[0])[::-1]
    print('User favourite cv&staff:')
    for ucs in user_cvstaff_sorted[:10]:
        if user_cvstaff[0, ucs] > 0:
            print(cvstaff_rev_map[ucs], user_cvstaff[0, ucs])

    # 根据协同过滤，近似用户推荐共同爱好
    sim_scores = np.matmul(user_vec, piv_norm.values)
    sim_scores_argsort = np.argsort(sim_scores)[0][::-1]
    topk_users = sim_scores_argsort[:topk]
    topk_mat = piv_norm.iloc[:, topk_users]
    topk_mat_values = topk_mat.values
    topk_index = np.argsort(topk_mat_values, axis=0)[::-1, :]
    # 选出这些用户喜欢的top10取并集
    topk_index = topk_index[:topk, :]
    fc_rec_list = collections.defaultdict(int)
    for i in range(topk_index.shape[0]):
        for j in range(topk_index.shape[1]):
            target_i = topk_index[i, j]
            if topk_mat_values[target_i, j] > 0 and piv_anime['type'][target_i] in valid_type:
                fc_rec_list[str(piv_anime['anime_id'][target_i])] += 1
    fc_rec_list = sorted([(k, piv_map[int(k)]['name'], v / topk) for k, v in fc_rec_list.items() if k not in user_info],
                         key=lambda x: x[-1], reverse=True)
    fc_rec_list = fc_rec_list[:return_num]

    # 根据官方tag获取类似作品
    tag_scores = np.matmul(tag_matrix, user_tags.transpose())[:, 0]  # [anime_num,1]->[anime_um,]
    tag_scores_argsort = np.argsort(tag_scores)[::-1]
    tag_rec_list = []
    idx = 0
    while len(tag_rec_list) < return_num and idx < len(tag_scores_argsort):
        target_i = tag_scores_argsort[idx]
        idx += 1
        target_id = str(piv_anime['anime_id'][target_i])
        if int(target_id) not in piv_map:
            continue
        if target_id in user_info or piv_map[int(target_id)]['type'] not in valid_type:
            continue
        target_name = piv_anime['name'][target_i]
        target_genre = piv_anime['genre'][target_i]
        target_score = tag_scores[target_i]

        tag_rec_list.append((target_id, target_name, target_score, target_genre))

    # 根据评论获取
    review_scores = np.matmul(review_mat, user_reviews.transpose())  # [anime_num, rev_rep]
    review_scores_argsort = np.argsort(review_scores, axis=0)[::-1, :]
    rev_rec_list = []
    rev_set = set()
    for i in range(rev_rep):
        idx = 0
        subidx = 0
        # 循环rev_rep次，每次
        while subidx < (return_num // rev_rep) and idx < len(review_scores_argsort):
            target_i = review_scores_argsort[idx, i]
            idx += 1
            target_id = str(review_rev_map[target_i].split('_')[0])
            if int(target_id) not in piv_map:
                continue
            if target_id in user_info or target_id in rev_set or piv_map[int(target_id)]['type'] not in valid_type:
                continue
            rev_set.add(target_id)
            subidx += 1
            target_name = review_rev_map[target_i]
            target_score = review_scores[target_i, i]
            rev_rec_list.append((target_id, target_name, target_score))

    # 获取cv，staff相关推荐
    cvstaff_scores = np.matmul(cvstaff_mat, user_cvstaff.transpose())[:, 0]  # [anime_num,1]->[anime_um,]
    cvstaff_scores_argsort = np.argsort(cvstaff_scores)[::-1]
    cvstaff_rec_list = []
    idx = 0
    while len(cvstaff_rec_list) < return_num and idx < len(cvstaff_scores_argsort):
        target_i = cvstaff_scores_argsort[idx]
        idx += 1
        target_name = cv_anime_rev_map[target_i]
        target_id = target_name.split('_')[0]
        if target_id in user_info:
            continue
        target_score = cvstaff_scores[target_i]
        target_cvs = cv_staff_data[target_name]['main'] + cv_staff_data[target_name]['sup']

        cvstaff_rec_list.append((target_id, target_name, target_score, target_cvs))

    res = {'fc_rec': fc_rec_list, 'tag_rec': tag_rec_list, 'rev_rec': rev_rec_list, 'cvstaff_rec': cvstaff_rec_list}

    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089)
