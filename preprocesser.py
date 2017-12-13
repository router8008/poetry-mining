from collections import Counter

import jieba.posseg as pseg
import os

import pickle


def cut_poetry(filename, saved_location):
    """
    对全唐诗分词
    :filename: 全唐诗输入文件位置
    :saved_location: 结果存储位置

    :return: list, 按顺序为
        char_counter：字频统计
        author_counter：作者计数
        word_set：词汇表
        word_counter：词汇计数
        word_property_dict：词汇词性
    """
    target_file_path = os.path.join(saved_location, 'cut_result.pkl')

    if os.path.exists(target_file_path) and os.path.exists(target_file_path):
        print('load existed cut result.')
        with open(target_file_path, 'rb') as f:
            result = pickle.load(f)
    else:
        char_counter = Counter()
        author_counter = Counter()
        word_set = set()
        word_counter = Counter()
        word_property_dict = {}

        line_cnt = 0
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == "":
                    continue
                if "【" in line:
                    header = line.split()[1]
                    author = header[header.find("】") + 1:].strip()
                    author_counter[author] += 1
                    continue
                characters = [c for c in line if '\u4e00' <= c <= '\u9fff']
                for char in characters:
                    char_counter[char] += 1
                cutted_line = pseg.cut(line)
                for word, property in cutted_line:
                    if not '\u4e00' <= word <= '\u9fff':
                        continue
                    word_property_dict[word] = property
                    word_set.add(word)
                    word_counter[word] += 1
                if line_cnt % 1000 == 0:
                    print('%d lines processed.' % line_cnt)
                line_cnt += 1
                if line_cnt > 2000:
                    break
        result = [char_counter, author_counter, word_set, word_counter, word_property_dict]
        with open(target_file_path, 'wb') as f:
            pickle.dump(result, f)
    return result


if __name__ == '__main__':
    saved_dir = os.curdir

    cut_poetry("全唐诗.txt", saved_dir)
