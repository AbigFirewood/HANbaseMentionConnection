import spacy

import torch

from torch.utils.data import DataLoader, Dataset


class DocumentsDataset(Dataset):
    def __init__(self, documents):
        super(DocumentsDataset, self).__init__()
        self.documents = documents  # 存储数据的

    def __len__(self):  # 返回长度
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    @staticmethod  # 获取train test 包装
    def build_train_test(train, test):
        return DocumentsDataset(train), DocumentsDataset(test)


class Vectorizer():  # 文本分词器对象
    def __init__(self, word_dict=None, max_sent_len=8, max_word_len=32):
        self.word_dict = word_dict  # 一个可选参数，用于传入一个词汇字典。如果没有传入，则默认为 None
        self.nlp = spacy.load("en_core_web_sm")
        # 使用 spacy 库加载了一个预训练的英文模型，并将其赋值给类的实例变量 self.nlp
        self.max_sent_len = max_sent_len  # 句子长度
        self.max_word_len = max_word_len  # 单词长度
        self.stop_words = None  # 停用词

    def vectorize_batch(self, t, trim=True):
        return self._vect_dict(t, trim)

    def _vect_dict(self, t, trim):  # 用于分词
        # 该方法接受两个参数：t（待处理的文本列表）和 trim（一个布尔值，指示是否需要对文本进行截断处理）。
        if self.word_dict is None:
            print(
                "单词表征文件缺失 \n 请检查文件设置 set a word_dict attribute \n first")
            raise Exception
        revs = []  # 用于存储处理后的文本列表。
        for rev in t:  # 遍历所有文本
            review = []  # 初始化处理后的文本
            for j, sent in enumerate(self.nlp(rev).sents):
                # 使用 self.nlp 方法（可能是某个自然语言处理工具或模型）将文本 rev 分割成句子/单词结构，并遍历每个句子。
                if trim and j >= self.max_sent_len:
                    # 如果 trim 为 True 且当前句子索引 j 大于或等于 self.max_sent_len（可能是类的一个属性，表示最大句子数量）
                    # 则停止处理更多句子。
                    break
                # 处理句子中的单词
                # 初始化处理结果
                s = []
                for k, word in enumerate(sent):  # 遍历单词结构
                    word = word.lower_  # 变小写
                    if trim and k >= self.max_word_len:  # trim表示如果单词超出数量是不是不再处理
                        break

                    if word in self.stop_words:  # 过滤停用词汇
                        continue
                    elif word in self.word_dict:  # 如果单词在 word_dict 中，则添加其对应的值到 s
                        s.append(self.word_dict[word])
                    else:
                        s.append(self.word_dict["_unk_word_"])  # _unk_word_
                if len(s) >= 1:
                    # 如果句子 s 包含至少一个单词，则将其转换为 PyTorch 的长整型张量并添加到 review 列表中。
                    review.append(torch.LongTensor(s))
            if len(review) == 0:
                # 如果 review 为空（即原始文本 rev 没有任何有效的单词或句子），则添加一个包含未知单词 _unk_word_ 的张量。
                review = [torch.LongTensor([self.word_dict["_unk_word_"]])]
            revs.append(review)
        # 返回处理后的文本列表 revs。 处理好的文本是 文本 句子 单词 三层结构 里面是文本在向量表中的标号！！不是向量表示
        return revs
