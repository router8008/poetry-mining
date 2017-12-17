import os

from analyzer import Analyzer
from preprocessor import CutResult, cut_poetry


def example():
    saved_dir = os.curdir
    result = cut_poetry("全唐诗.txt", saved_dir)
    a = Analyzer(result, saved_dir)
    # print(a.find_similar_poet("李世民"))


if __name__ == '__main__':
    example()

    # tsne(word_vector, authors)
