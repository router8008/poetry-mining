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


def get_word_vector(author_poetry_dict):
    """
    解析词向量
    :return:authors: 作者列表
            word_vector: 对应的词向量
    """
    authors = list(author_poetry_dict.keys())
    poetry = list(author_poetry_dict.values())
    vectorizer = CountVectorizer(min_df=1)
    word_matrix = vectorizer.fit_transform(poetry).toarray()
    transformer = TfidfTransformer()
    word_vector = transformer.fit_transform(word_matrix).toarray()
    return word_vector, authors


# mpl.rcParams['font.sans-serif'] = ['AR PL UMing CN']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


def plot_vectors(X, target):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], target[i],
                 # color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 8}
                 , fontproperties=font
                 )


def tsne(word_vector, authors):
    t_sne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = t_sne.fit_transform(word_vector)
    plot_vectors(x_tsne, authors)
    plt.show()


if __name__ == '__main__':
    saved_dir = os.curdir

    result = cut_poetry("全唐诗.txt", saved_dir)
    word_vector, authors = get_word_vector(result.author_poetry_dict)
    tsne(word_vector, authors)
