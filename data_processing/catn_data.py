import os, sys, gzip
import argparse, random, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from operator import itemgetter
from collections import defaultdict
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import json


class CrossData:
    def __init__(self, path_s, path_t, ratio):
        self.path_s, self.path_t, self.ratio= path_s, path_t, ratio
        self.df_s, self.df_t = self.get_df()
        self.udict, self.idict_s, self.idict_t = self.convert_idx()
        self.coldstart_user_set, self.common_user_set, self.train_common_s, self.train_common_t, \
            self.train_s, self.train_t, self.train_cs_s, self.test, self.test_new, self.test_none, self.test_tail, self.common_user_all_set = self.split_train_test()
        self.vocab_dict, self.word_embedding = self.get_w2v()
        self.docu_udict_s, self.docu_idict_s, self.auxiliary_docu_udict_s = \
            self.get_documents(self.train_s, self.coldstart_user_set | self.common_user_set, 0)
        self.docu_udict_t, self.docu_idict_t, self.auxiliary_docu_udict_t = \
            self.get_documents(self.train_t, self.common_user_set, 1)

    def get_df(self):
        def parse(path):
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    yield json.loads(line)

        def get_raw_df(path):
            df = {}
            for i, d in tqdm(enumerate(parse(path)), ascii=True):
                df[i] = d
            df = pd.DataFrame.from_dict(df, orient='index')
            return df

        csv_path_s = self.path_s.replace('.jsonl', '.csv')
        csv_path_t = self.path_t.replace('.jsonl', '.csv')

        if os.path.exists(csv_path_s) and os.path.exists(csv_path_t):
            df_s = pd.read_csv(csv_path_s)
            df_t = pd.read_csv(csv_path_t)
            print('Load raw data from %s.' % csv_path_s)
            print('Load raw data from %s.' % csv_path_t)
        else:
            df_s = get_raw_df(self.path_s)
            df_t = get_raw_df(self.path_t)

            df_s.to_csv(csv_path_s, index=False)
            df_t.to_csv(csv_path_t, index=False)
            print('Build raw data to %s.' % csv_path_s)
            print('Build raw data to %s.' % csv_path_t)

        return df_s, df_t


    def convert_idx(self):
        uiterator = count(0)
        udict = defaultdict(lambda: next(uiterator))
        [udict[user] for user in self.df_s["reviewerID"].tolist()+ self.df_t["reviewerID"].tolist()]
        iiterator_s = count(0)
        idict_s = defaultdict(lambda: next(iiterator_s))
        [idict_s[item] for item in self.df_s["asin"]]
        iiterator_t = count(0)
        idict_t = defaultdict(lambda: next(iiterator_t))
        [idict_t[item] for item in self.df_t["asin"]]


        user_set_s = set(self.df_s['reviewerID'])
        item_set_s = set(self.df_s['asin'])
        user_set_t = set(self.df_t['reviewerID'])
        item_set_t = set(self.df_t['asin'])
        all_user_set = user_set_s | user_set_t

        self.user_num_s, self.item_num_s, self.user_num_t, self.item_num_t, self.user_num = \
            len(user_set_s), len(item_set_s), len(user_set_t), len(item_set_t), len(all_user_set)

        print('Source domain users %d, items %d, ratings %d.' % (self.user_num_s, self.item_num_s, len(self.df_s)))
        print('Target domain users %d, items %d, ratings %d.' % (self.user_num_t, self.item_num_t, len(self.df_t)))

        return dict(udict), dict(idict_s), dict(idict_t)
    

    def split_train_test(self):
        src_users = set(self.df_s.reviewerID.unique())
        tgt_users = set(self.df_t.reviewerID.unique())
        co_users = src_users & tgt_users

        co_users = sorted(list(co_users))
        random.seed(2024)  # 设置固定的随机种子
        test_users = random.sample(co_users, round(self.ratio * len(co_users)))

        test_df2 = self.df_t[self.df_t['reviewerID'].isin(test_users)]

        # test_new = test[test['iid'].isin(set(train_common_t.iid.unique()))]
        # test_df2_new = test_df2[test_df2['asin'].isin(all_train_data2['parent_asin'].unique())]
        # pkl_path = os.path.dirname(self.path_s)
        # test_df2.to_csv(os.path.join(pkl_path, 'domain2_test_data.csv'), index=False)

        self.df_s['uid'] = self.df_s['reviewerID'].map(lambda x: self.udict[x])
        self.df_t['uid'] = self.df_t['reviewerID'].map(lambda x: self.udict[x])
        self.df_s['iid'] = self.df_s['asin'].map(lambda x: self.idict_s[x])
        self.df_t['iid'] = self.df_t['asin'].map(lambda x: self.idict_t[x])

        coldstart_user_set = set(self.udict[user_id] for user_id in test_users)
        overlap_user_set = set(self.udict[user_id] for user_id in co_users)
        # common_user_all_set = overlap_user_set-coldstart_user_set
        # common_user_set = common_user_all_set

        other_user_set = set(self.df_t.uid.unique()) - overlap_user_set

        common_user_set = overlap_user_set-coldstart_user_set
        common_user_all_set = common_user_set | other_user_set

        # 实际用于训练
        train_common_s = self.df_s[self.df_s['uid'].isin(common_user_set)]
        train_common_t = self.df_t[self.df_t['uid'].isin(common_user_set)]
        # 用于计算物品平均分数
        train_s = self.df_s
        train_t = self.df_t[self.df_t['uid'].isin(coldstart_user_set).apply(lambda x: not x)]
        train_cs_s = self.df_s[self.df_s['uid'].isin(coldstart_user_set)]

        test = self.df_t[self.df_t['uid'].isin(coldstart_user_set)]
        test_new = test[test['iid'].isin(set(train_t.iid.unique()))]
        test_none = test[~test['iid'].isin(set(train_t.iid.unique()))]
        item_counts = train_t['iid'].value_counts()
        tail_items = item_counts[item_counts < 10].index
        test_tail = test[test['iid'].isin(tail_items)]

        # pkl_path = os.path.dirname(self.path_s)
        # test.to_csv(os.path.join(pkl_path, 'domain2_test_data.csv'), index=False)
        # test_new.to_csv(os.path.join(pkl_path, 'domain2_test_new.csv'), index=False)
        # test_none.to_csv(os.path.join(pkl_path, 'domain2_test_none.csv'), index=False)
        # test_tail.to_csv(os.path.join(pkl_path, 'domain2_test_tail.csv'), index=False)


        return coldstart_user_set, common_user_set, train_common_s, train_common_t, train_s, train_t, train_cs_s, test, test_new, test_none, test_tail, common_user_all_set

    def get_w2v(self):
        pkl_path = os.path.dirname(self.path_s)
        vocab_dict_path = os.path.join(pkl_path, 'vocab_dict.pkl')
        word_embedding_path = os.path.join(pkl_path, 'word_embedding.npy')

        if not os.path.exists(vocab_dict_path) or not os.path.exists(word_embedding_path):
            all_text = self.df_s['reviewText'].tolist()
            all_text.extend(self.df_t['reviewText'].tolist())
            vectorizer = TfidfVectorizer(max_df=0.5, stop_words='english', max_features=20000)
            tfidf = vectorizer.fit_transform(all_text)
            vocab_dict = vectorizer.vocabulary_

            word_embedding = np.zeros([len(vocab_dict) + 1, 300])

            print('Building word embedding matrix...')
            for word, idx in vocab_dict.items():
                if word in google_model.key_to_index:
                    word_embedding[idx] = google_model[word]

            with open(vocab_dict_path, 'wb') as f:
                pickle.dump(vocab_dict, f)
            np.save(word_embedding_path, word_embedding)
        else:
            with open(vocab_dict_path, 'rb') as f:
                vocab_dict = pickle.load(f)
            word_embedding = np.load(word_embedding_path)
        
        print('Load word_embedding ')

        return vocab_dict, word_embedding


    def get_documents(self, df, user_set, flag):
        print('in get_documents')
        reviews = [list(map(lambda x: self.vocab_dict.get(x, -1), review.split(' '))) for review in df['reviewText']]
        reviews = [np.array(review)[np.array(review) != -1].tolist() for review in reviews]
        df = df.copy()
        df['review_idx'] = reviews
        max_length = 50
        
        
        pkl_path = os.path.dirname(self.path_s)
        auxiliary_docu_udict_path = os.path.join(pkl_path, str(flag)+'auxiliary_docu_udict.pkl')
        cut_auxiliary_docu_udict_path = os.path.join(pkl_path, str(flag)+'cut_auxiliary_docu_udict.pkl')
        if not os.path.exists(auxiliary_docu_udict_path) or not os.path.exists(cut_auxiliary_docu_udict_path):
            auxiliary_docu_udict, cut_auxiliary_docu_udict = defaultdict(list), defaultdict(list)
            print('Constructing auxiliary documents...')
            df_aux = df[~df['uid'].isin(user_set)]
            item_rating_to_reviews = df_aux.groupby(['iid', 'overall'])['review_idx'].apply(list).to_dict()

            for idx, row in tqdm(df_aux.iterrows(), ascii=True, total=df_aux.shape[0]):
                user, item, rating = row['uid'], row['iid'], row['overall']
                for offset in [0, 1, -1]:
                    reviews = item_rating_to_reviews.get((item, rating + offset))
                    if reviews:
                        auxiliary_docu_udict[user].extend(random.choice(reviews))
                        break
            
            for u, docu in auxiliary_docu_udict.items():
                docu_cut = np.array(docu)[:max_length]
                cut_auxiliary_docu_udict[u] = np.pad(docu_cut, (0, max_length - docu_cut.shape[0]),
                                                    'constant', constant_values=(0, -1))
            with open(auxiliary_docu_udict_path, 'wb') as f:
                pickle.dump(auxiliary_docu_udict, f)
            with open(cut_auxiliary_docu_udict_path, 'wb') as f:
                pickle.dump(cut_auxiliary_docu_udict, f)
        else:
            # 加载
            with open(auxiliary_docu_udict_path, 'rb') as f:
                auxiliary_docu_udict = pickle.load(f)
            with open(cut_auxiliary_docu_udict_path, 'rb') as f:
                cut_auxiliary_docu_udict = pickle.load(f)

        
        docu_udict, cut_docu_udict = defaultdict(list), defaultdict(list)
        docu_idict, cut_docu_idict = defaultdict(list), defaultdict(list)

        for user, item, review in zip(df['uid'], df['iid'], df['review_idx']):
            docu_udict[user].extend(review)
            docu_idict[item].extend(review)

        
        for u, docu in docu_udict.items():
            docu_cut = np.array(docu)[:max_length]
            cut_docu_udict[u] = np.pad(docu_cut, (0, max_length - docu_cut.shape[0]),
                                       'constant', constant_values=(0, -1))
        for i, docu in docu_idict.items():
            docu_cut = np.array(docu)[:max_length]
            cut_docu_idict[i] = np.pad(docu_cut, (0, max_length - docu_cut.shape[0]),
                                       'constant', constant_values=(0, -1))
        
        return cut_docu_udict, cut_docu_idict, cut_auxiliary_docu_udict


    def dump_pkl(self):
        def extract_ratings(df):
            ratings = df.apply(lambda x:(x['uid'], x['iid'], x['overall']), axis=1).tolist()
            return ratings

        pkl_path = self.path_s.replace(self.path_s.split('/')[-1], 'crossdata_%.2f.pkl' % (self.ratio))
        with open(pkl_path, 'wb') as f:
            data = [self.udict, self.idict_s, self.idict_t, self.coldstart_user_set, self.common_user_set,
                    self.user_num, self.item_num_s, self.item_num_t,
                    extract_ratings(self.train_common_s), extract_ratings(self.train_common_t),
                    extract_ratings(self.train_cs_s), extract_ratings(self.test),
                    extract_ratings(self.test_new), extract_ratings(self.test_none), extract_ratings(self.test_tail),
                    extract_ratings(self.train_s), extract_ratings(self.train_t),
                    self.vocab_dict, self.word_embedding,
                    self.docu_udict_s, self.docu_idict_s, self.docu_udict_t, self.docu_idict_t,
                    self.auxiliary_docu_udict_s, self.auxiliary_docu_udict_t
                    ]
            pickle.dump(data, f)
            print('Build data to %s.' % pkl_path)
            print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--info', type=bool, default=True)
    args = parser.parse_args()

    if not args.info:
        print('Loading GoogleNews w2v model...')
        google_model = KeyedVectors.load_word2vec_format('/root/autodl-tmp/catn/GoogleNews-vectors-negative300.bin', binary=True)
    
    path = '/root/autodl-tmp/catn/'
    CrossData(path + 'movie2music/movie.jsonl', path + 'movie2music/music.jsonl', ratio=args.ratio).dump_pkl()
    # CrossData(path + 'book2movie/book.jsonl', path + 'book2movie/movie.jsonl', ratio=args.ratio).dump_pkl()
    # CrossData(path + 'book2music/book.jsonl', path + 'book2music/music.jsonl', ratio=args.ratio).dump_pkl()