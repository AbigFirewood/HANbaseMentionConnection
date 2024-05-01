import json
import time

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from component.DataHelper import DocumentsDataset, Vectorizer
from component.Nets import HANConnect
import logging


def get_data_from_json(file):  # 读取json文件的函数
    f = open(file, "r")
    content = f.read()
    a = json.loads(content)
    # print(type(a["documents"]))
    return a["documents"]


def get_train_and_test_from_json(train, test):  # 用于获取数据
    print("\n基本数据加载:\n" + 25 * "-")
    return DocumentsDataset.build_train_test(get_data_from_json(train), get_data_from_json(test))


def get_word_tenser(string):
    get = string.split(',')
    # print(get)
    for i in range(len(get)):
        get[i] = float(get[i])
    t = torch.Tensor(get)
    print(t)


def load_embeddings(file):  # 加载HAN词嵌入表
    emb_file = open(file, "r", encoding='utf-8').readlines()
    word_d = {"_padding_": 0, "_unk_word_": 1}  # 初始化字典
    # 加载词汇表
    word_t = []
    for i, line in tqdm(enumerate(emb_file, 2), desc="加载词嵌入表中...", total=len(emb_file)):
        sql = line.strip().split()
        word_d[sql[0]] = i
        # 将 spl 列表的第一个元素（即单词）作为键 将当前索引 i 作为值，存储到字典 word_d 中。
        # 这样，word_d 字典就建立了一个从单词到其在 tensor 中位置的映射。
        # 取出文件中的词向量
        word_emb_list = sql[2].split(',')
        # 转化float
        for j in range(len(word_emb_list)):
            word_emb_list[j] = float(word_emb_list[j])
        # 加入词汇表
        word_t.append(word_emb_list)
    # tensor产生
    word_t = torch.Tensor(word_t)
    # print(t.shape) 加入pad
    zero = torch.zeros(1, word_t.shape[1], dtype=torch.float32)
    word_t = torch.cat((zero, word_t), dim=0)
    word_t = torch.cat((zero, word_t), dim=0)
    # print(word_d)
    try:
        assert (len(word_d) == word_t.shape[0])
        # 使用断言来检查 word_d 字典的长度是否与预期的（词的数量 + 2）相同。如果不相同，则打印错误消息。
    except:
        print("词向量加载失败！！！")
    # 返回填充好的张量 tensor 和词到索引的映射字典 word_d。
    return word_t, word_d


def load_stop_words(file):  # 加载停用词表
    stop_file = open(file, "r", encoding="utf-8").readlines()
    for i, _ in tqdm(enumerate(stop_file), desc="加载停用词表中...", total=len(stop_file)):  # range(len(stop_file)):
        stop_file[i] = stop_file[i].strip()  # 去掉换行符
    return stop_file


def load_ent_vec(file):  # 加载实体嵌入
    ent_dic = {}
    ent_file = open(file, "r", encoding="utf-8").readlines()
    for i, ent in tqdm(enumerate(ent_file), desc="加载实体表征中...", total=len(ent_file)):
        ent = ent.strip().split()
        ent_v = ent[1].split(',')
        for j in range(len(ent_v)):
            ent_v[j] = float(ent_v[j])
        ent_dic[ent[0]] = torch.Tensor(ent_v)
    return ent_dic


def get_mentions_in_sent(start, sent_index, mentions):  # 获取对应句子的所有mention信息
    sent_m = []
    next_n = 0
    for m_n, mention in enumerate(mentions[start:], start):  # 左闭右开
        if mention["sent_index"] > sent_index:
            next_n = m_n
            break
        if mention["sent_index"] == sent_index:  # 如果发现句子匹配
            # sent_m.append([mention["gold_index"]]) # 加入值1
            candidates = mention["candidates"].split("\t")
            can_l = []  # 一个mention的候选者
            for i in range(len(candidates)):  # 获取所有candidates的标号
                if i % 2 == 0:
                    continue
                can_l.append(candidates[i])
            next_n = m_n + 1  # 更新next_n
            sent_m.append((can_l, mention["gold_index"], sent_index))

    return sent_m, next_n
    # 返回 结构是 （所有备选,gode,对应句子id）元组的list 和下一个切片的起始位置


def get_stat(list_doc, mentions):  # 获取stat，包含一系列以句子为单位的统计量
    stat = []
    for doc_n, doc in enumerate(list_doc):
        start_index = 0
        for sent_n, sent in enumerate(list_doc[doc_n]):
            sent_mentions, start_index = get_mentions_in_sent(start_index, sent_n, mentions[doc_n])
            stat.append((len(sent), len(doc), doc_n, sent_n, sent, sent_mentions))
    return sorted(stat, reverse=True)
    # stat结构：按照句子长度排序的(句子长度,文档长度,文档标号,句子在文档中的标号,句子原文,[([str :所有备选标号],对应正确标号,对应句子在文档中的标号)])


def batcher_builder(vectorizer, trim=True):
    def doc_batch(dic):
        # 这是 tuple_batcher_builder 返回的内部函数，它接受一个字典dic，包括所有信息
        document = []
        mentions = []
        # 将列表 l 的元素解压为两个独立的列表doc 和 men
        for i in range(len(dic)):
            document.append(dic[i]["document"])
            mentions.append(dic[i]["mentions"])
        list_doc = vectorizer.vectorize_batch(document, trim)  # 进行分词处理
        stat = get_stat(list_doc, mentions)  # list_doc : [[tensor{},tensor{}...]...]
        # stat结构：按照句子长度排序的(句子长度,文档长度,文档标号,句子在文档中的标号,句子原文,[([所有备选标号],对应正确标号,对应句子在文档中的标号)])
        max_len = stat[0][0]  # 找到最长的哪一个句子的长度
        batch_t = torch.zeros(len(stat), max_len, dtype=torch.long)
        # 一句话一个向量
        for i, s in enumerate(stat):
            for j, w in enumerate(s[-2]):  # s[-2] is sentence in stat tuple
                batch_t[i, j] = w  # 遍历 stat 列表，将每个句子中的每个单词填充到 batch_t 张量的相应位置 里面是文本在向量表中的标号！！不是向量表示
        stat = [(ls, lr, r_n, s_n, s_m) for ls, lr, r_n, s_n, _, s_m in stat]  # 去掉了原文表示 没必要再存储一份
        # 返回 处理好的句子表示 [句子[单词序号]]
        # [[单词标号]] 统计数据 排序的(句子长度,文档长度,文档标号,句子在文档中的标号,[([str:所有备选标号],对应正确标号,对应句子在文档中的标号)])
        # 和原来的文档结构
        return batch_t, stat, document

    return doc_batch


def get_loss(criterion, out, gold_list, mask_list):  # [tensor[a,b,c.....],int]
    device = out.device
    target = torch.zeros(out.shape, device=device)
    for i, gold in enumerate(gold_list):
        target[i, :mask_list[i]] = out[i, gold - 1]  # 使用gold填充target
        target[i, mask_list[i]:] = out[i, -1]  # 对于pad的部分填充最后一个pad元素
    flag = torch.ones(out.shape, device=device)  # 全部是1代表正向排序
    loss = criterion(target, out, flag)
    return loss


def accuracy(out, gold_list):  # [tensor[a,b,c.....],int]
    all_acc = 0
    _, max_i = torch.max(out, 1)  # 找到最大值对应的index
    for i, men in enumerate(gold_list):
        if max_i[i].item() == men - 1:
            all_acc += 1
    temp = torch.Tensor([all_acc]).float()
    int = len(out)
    return all_acc, temp / len(out) * 100


def log_helper(file):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def train(epoch, epochs, net, optimizer, dataset, criterion, device, logger):
    """
    :param logger:
   :param epoch: 当前周期
   :param epochs: 全部周期
   :param net: 神经网络
   :param optimizer: 优化器
   :param dataset: 数据集迭代器
   :param criterion: 目标函数
   :param device: 设备
   :return:
   """
    epoch_loss = 0
    ok_all = 0
    data_tensors = torch.LongTensor().to(device)  # 准备数据tenser
    # epoch_loss: 用于累计当前epoch的损失值。
    # ok_all: 用于累计当前epoch中预测正确的样本数。
    # data_tensors: 使用new_tensors函数创建新的张量（为批处理数据准备的）。
    with tqdm(total=len(dataset), desc="训练中:Epoch {}/{}".format(epoch, epochs)) as pbar:
        for iteration, (batch_t, stat, document) in enumerate(dataset):
            data = data_tensors.resize_(batch_t.size()).copy_(batch_t)
            optimizer.zero_grad()  # 清除之前的梯度
            # for name, param in net.named_parameters():
            '''
            # print(name, param.size())
                print(name)  # , param.size())
                print(param.grad)  # 打印权重梯度
            '''
            out, gold_list, mask_list = net(data, stat)  # 前向传播
            loss = get_loss(criterion, out, gold_list, mask_list)  # tensor[loss]
            ok, per = accuracy(out, gold_list)
            # 累计当前批次的损失值
            epoch_loss += loss.item()
            # 执行反向传播，计算梯度
            loss.backward()
            '''
            for name, param in net.named_parameters():
                # print(name, param.size())
                print(name)  # , param.size())
                print(param.grad)  # 打印权重梯度
            '''
            optimizer.step()  # 优化
            ok_all += per.item()
            # 使用优化器更新模型的参数
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({"acc": ok_all / (iteration + 1), "CE": epoch_loss / (iteration + 1)})
        logger.info("===> Epoch {}/{} Complete: Avg. Loss: {:.4f}, {}% accuracy".format(epoch, epochs,
                                                                                        epoch_loss / len(dataset),
                                                                                        ok_all / len(dataset)))


def test(epoch, epochs, net, dataset, criterion, device, logger, max_acc):
    epoch_loss = 0
    ok_all = 0
    pred = 0
    with torch.no_grad():
        data_tensors = torch.LongTensor().to(device)  # 准备数据tenser
        with tqdm(total=len(dataset), desc="测试中:Epoch {}/{}".format(epoch + 1, epochs)) as pbar:
            for iteration, (batch_t, stat, dcoument) in enumerate(dataset):
                data = data_tensors.resize_(batch_t.size()).copy_(batch_t)
                out, gold_list, mask_list = net(data, stat)  # (评分tensor,gold）list 前向传播
                loss = get_loss(criterion, out, gold_list, mask_list)
                ok, per = accuracy(out, gold_list)
                epoch_loss += loss.item()
                ok_all += per.data[0]
                pred += 1
                pbar.update(1)
                pbar.set_postfix({"acc": ok_all / pred})
        max_acc = max(max_acc, ok_all / pred)  # 更新最大准确
    logger.info("===> TEST Complete:  Avg. Loss: {:.4f}, {}% accuracy {}% max_acc".format(epoch_loss / len(dataset),
                                                                                          ok_all / pred, max_acc))
    return max_acc


def main():
    print(32 * "-" + "\n基于HAN的文本对齐网络:\n" + 32 * "-")

    hid_size = 300  # 设置表征维度
    batch_size = 4
    learning_rate = 1e-5
    epochs = 500
    num_workers = 0  # 2  # 0  # 2  # 0  # 2
    clip_grad = 10

    # 路径
    file_test = ".//dataset//documents_test.json"  # 路径
    file_train = ".//dataset//documents_train.json"
    file_word_info = ".//dataset//word_info.txt"
    file_stop_word = ".//dataset//stopword.txt"
    file_ent_vec = ".//dataset//ent_vec.txt"
    file_log = ".//log//log.txt"
    logger = log_helper(file_log)

    # 分词器设置
    max_words = 16
    max_sents = 32

    # 设备
    device = 'cpu'  # cuda' if torch.cuda.is_available() else 'cpu'

    # 获取基本数据
    train_set, test_set = get_train_and_test_from_json(file_train, file_test)
    print("Train set length:", len(train_set))
    print("Test set length:", len(test_set))
    # print(test_set[0]["document"])

    # 加载HAN词嵌入表
    word_tensor, word_dic = load_embeddings(file_word_info)
    # 加载停用词表
    stop_word = load_stop_words(file_stop_word)
    # 加载实体矩阵特征向量表
    ent_dic = load_ent_vec(file_ent_vec)
    # print(len(ent_dic['25493']))

    # 加载分词器
    vectorizer = Vectorizer(max_word_len=max_words, max_sent_len=max_sents)

    # 加载模型：
    net = HANConnect(ntoken=len(word_dic), emb_size=len(word_tensor[1]), hid_size=hid_size)
    # 传入参数解释 ： ntoken 单词数量 emb_size 是词向量的长度 hid_size是AttentionalBiGRU的层数
    # 设置词嵌入向量tensor
    net.set_emb_tensor(torch.FloatTensor(word_tensor))
    net.set_ent_dic(ent_dic)
    # 设置模型的词嵌入字典
    vectorizer.word_dict = word_dic
    vectorizer.stop_words = stop_word
    # 将加载的词典赋值给vectorized的词典属性。

    train_batch_builder = batcher_builder(vectorizer, trim=True)  # 获取batch处理函数
    test_batch_builder = batcher_builder(vectorizer, trim=True)  # 获取batch处理函数

    # 加载数据加载助手
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=train_batch_builder, pin_memory=True)
    dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 collate_fn=test_batch_builder, pin_memory=True)

    # 设置优化目标函数
    criterion = nn.MarginRankingLoss(reduction='sum')  # reduction='mean')

    # 加载模型到设备
    net = net.to(device)
    # print("-" * 20)
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 使用随机梯度下降（SGD）作为优化方法，并且学习率（learning rate）设置为1e-5
    nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
    start = time.perf_counter()  # 记录时间开始
    logger.info(32 * "-" + "program start!" + 32 * "-")
    max_test_acc = -1.0
    for epoch in range(1, epochs + 1):
        logger.info(32 * "-" + "epoch{}start!".format(epoch) + 32 * "-")
        train(epoch, epochs + 1, net, optimizer, dataloader_train, criterion, device, logger)  # 训练
        max_test_acc = test(epoch, epochs + 1, net, dataloader_test, criterion, device, logger, max_test_acc)  # 测试
        logger.info(32 * "-" + "epoch{}end!".format(epoch) + 32 * "-")
    end = time.perf_counter()
    logger.info(32 * "-" + "The program ends in {} s".format(str(end - start)) + 32 * "-")
    logger.info("The max_test_acc is {}".format(max_test_acc))
    logger.info("device" + str(torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'))


if __name__ == '__main__':  # 执行入口
    main()
