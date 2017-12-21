import os

from analyzer import Analyzer, plot_vectors
from preprocessor import CutResult, cut_poetry


def print_counter(counter):
    for key, value in counter:
        print(key, value)
    print()


def example():
    saved_dir = os.path.join(os.curdir, "out")
    result = cut_poetry("全唐诗.txt", saved_dir)
    analyzer = Analyzer(result, saved_dir)
    # 画图
    tf_idf_vector_list = []
    w2v_vector_list = []
    author_list = []
    for c in result.author_counter.most_common(100):
        author = c[0]
        index = analyzer.authors.index(author)
        w2v_vector_list.append(analyzer.w2v_word_vector_tsne[index])
        tf_idf_vector_list.append(analyzer.tfidf_word_vector_tsne[index])
        author_list.append(author)
    plot_vectors(tf_idf_vector_list, author_list)
    plot_vectors(w2v_vector_list, author_list)

    print("**基于统计的分析")
    print("写作数量排名：")
    print_counter(result.author_counter.most_common(10))

    print("最常用的词：")
    cnt = 0
    l = []
    for word, count in result.word_counter.most_common():
        if cnt == 10:
            break
        if len(word) > 1:
            l.append((word, count))
            cnt += 1
    print_counter(l)

    print("最常用的名词：")
    print_counter(result.word_property_counter_dict['n'].most_common(10))

    print("最常见的地名：")
    print_counter(result.word_property_counter_dict['ns'].most_common(10))

    print("最常见的形容词：")
    print_counter(result.word_property_counter_dict['a'].most_common(10))

    print("**基于词向量的分析")
    for word in ["春", "鸳鸯", "垂柳", "枕"]:
        print("与 %s 相关的词：" % word)
        print_counter(analyzer.find_similar_word(word))

    for poet in ["李白", "杜甫", "白居易"]:
        print("与 %s 用词相近的诗人：" % poet)
        print("根据tf-idf标准： %s" % analyzer.find_similar_poet(poet))
        print("根据word2vector标准： %s\n" % analyzer.find_similar_poet(poet, use_w2v=True))


if __name__ == '__main__':
    example()
