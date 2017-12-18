import os

from analyzer import Analyzer, plot_vectors
from preprocessor import CutResult, cut_poetry


def print_counter(counter):
    for key, value in counter:
        print(key, value)
    print()


def example():
    saved_dir = os.curdir
    result = cut_poetry("全唐诗.txt", saved_dir)
    analyzer = Analyzer(result, saved_dir)
    # plot_vectors(analyzer.w2v_word_vector_tsne, analyzer.analyzer.thors)

    # 基于统计的分析
    print("唐诗写作数量：")
    print_counter(result.author_counter.most_common(10))

    print("常用字排名：")
    print_counter(result.word_counter.most_common(10))

    print("最常见的人名：")
    print_counter(result.word_property_counter_dict['nr'].most_common(10))

    print("最常见的地名：")
    print_counter(result.word_property_counter_dict['ns'].most_common(10))

    print("最常见的形容词：")
    print_counter(result.word_property_counter_dict['a'].most_common(10))

    print("最常见的成语：")
    print_counter(result.word_property_counter_dict['i'].most_common(10))

    print("最常见的语气词：")
    print_counter(result.word_property_counter_dict['y'].most_common(10))

    # 基于词向量的分析
    for word in ["春", "鸳鸯", "垂柳", "枕"]:
        print("与 %s 相关的词：" % word)
        print_counter(analyzer.find_similar_word(word))

    for poet in ["李白", "杜甫", "白居易"]:
        print("与 %s 用词相近的诗人：" % poet)
        print("根据tf-idf标准： %s" % analyzer.find_similar_poet(poet))
        print("根据word2vector标准： %s\n" % analyzer.find_similar_poet(poet, use_w2v=True))


if __name__ == '__main__':
    example()
