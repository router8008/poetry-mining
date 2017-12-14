import os
import pickle

from collections import Counter, OrderedDict
from jieba import posseg as pseg


class CuttedResult(object):
    """
    分词结果
    char_counter：字频统计
    author_counter：作者计数
    word_set：词汇表
    word_counter：词汇计数
    word_property_dict：词汇词性
    author_poetry_dict：分词后的结果
    """

    def __init__(self):
        self.char_counter = Counter()
        self.author_counter = Counter()
        self.word_set = set()
        self.word_counter = Counter()
        self.word_property_dict = {}
        self.author_poetry_dict = OrderedDict()

    def add_cut_poetry(self, author, divided_lines):
        """
        为author_poetry_dict添加对象
        """
        ctp = self.author_poetry_dict.get(author)
        if ctp is None:
            self.author_poetry_dict[author] = ""
        else:
            self.author_poetry_dict[author] += ' '
        self.author_poetry_dict[author] += ' '.join(divided_lines)


def is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'


def cut_poetry(filename, saved_location):
    """
    对全唐诗分词
    :filename: 全唐诗输入文件位置
    :saved_location: 结果存储位置
    :return: CuttedResult
    """
    target_file_path = os.path.join(saved_location, 'cut_result.pkl')
    if os.path.exists(target_file_path) and os.path.exists(target_file_path):
        print('load existed cut result.')
        with open(target_file_path, 'rb') as f:
            result = pickle.load(f)
    else:
        result = CuttedResult()
        line_count = 0
        current_author = None
        divided_lines = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == "":
                    continue
                # 解析作者
                if "【" in line:
                    header = line.split()[1]
                    author = header[header.find("】") + 1:].strip()
                    result.author_counter[author] += 1
                    # 将当前分词后的结果加入结果表中
                    if current_author is not None:
                        result.add_cut_poetry(current_author, divided_lines)
                        divided_lines = []
                    current_author = author
                    continue
                # 解析诗句
                chars = [c for c in line if is_chinese(c)]
                for char in chars:
                    result.char_counter[char] += 1
                cut_line = pseg.cut(line)
                for word, property in cut_line:
                    if not is_chinese(word):
                        continue
                    result.word_property_dict[word] = property
                    result.word_set.add(word)
                    result.word_counter[word] += 1
                    divided_lines.append(word)
                if line_count % 1000 == 0:
                    print('%d lines processed.' % line_count)
                line_count += 1
                if line_count > 2000:
                    break
        with open(target_file_path, 'wb') as f:
            pickle.dump(result, f)
    return result
