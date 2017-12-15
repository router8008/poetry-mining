# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from preprocessor import CutResult, cut_poetry
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import manifold

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"/usr/local/share/fonts/simhei.ttf", size=14)

# mpl.rcParams['font.sans-serif'] = ['AR PL UMing CN']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


class Analyzer(object):
    """
    authors: 作者列表
    word_vector: 对应的词向量
    """

    def __init__(self, cut_result):
        self.authors, self.word_vector = self.author_word_vector(cut_result.author_poetry_dict)
        self.word_vector_tsne = self.tsne()

    @staticmethod
    def author_word_vector(author_poetry_dict):
        """解析每个作者的词向量"""
        authors = list(author_poetry_dict.keys())
        poetry = list(author_poetry_dict.values())
        vectorizer = CountVectorizer(min_df=1)
        word_matrix = vectorizer.fit_transform(poetry).toarray()
        transformer = TfidfTransformer()
        word_vector = transformer.fit_transform(word_matrix).toarray()
        return authors, word_vector

    def tsne(self):
        t_sne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        word_vector_tsne = t_sne.fit_transform(self.word_vector)
        return word_vector_tsne

    def find_similar_poet(self, poet_name):
        """
        通过词向量寻找最相似的诗人
        :param: poet: 需要寻找的诗人名称
        :return:最匹配的诗人
        """
        poet_index = self.authors.index(poet_name)
        x = self.word_vector[poet_index]
        min_angle = np.pi
        min_index = 0
        for i, author in enumerate(self.authors):
            if i == poet_index:
                continue
            y = self.word_vector[i]
            cos = x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))
            angle = np.arccos(cos)
            if min_angle > angle:
                min_angle = angle
                min_index = i
        return self.authors[min_index]


def plot_vectors(X, target):
    """绘制结果"""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], target[i],
                 # color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 8}
                 , fontproperties=font
                 )
