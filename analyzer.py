import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from preprocessor import CuttedResult, cut_poetry


def get_author_word_vector(author_poetry_dict):
    """
    解析词向量
    :return:
    """
    authors = list(author_poetry_dict.keys())
    poetry = list(author_poetry_dict.values())
    vectorizer = CountVectorizer(min_df=1)
    word_matrix = vectorizer.fit_transform(poetry).toarray()
    transformer = TfidfTransformer()
    word_vector = transformer.fit_transform(word_matrix).toarray()
    author_word_vector = {}
    for i in range(len(authors)):
        author_word_vector[authors[i]] = word_vector[i]
    return author_word_vector


if __name__ == '__main__':
    saved_dir = os.curdir

    result = cut_poetry("全唐诗.txt", saved_dir)
    get_author_word_vector(result.author_poetry_dict)
