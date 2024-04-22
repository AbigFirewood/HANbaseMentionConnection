import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict


class AttentionalBiGRU(nn.Module):

    def __init__(self, inp_size, hid_size, dropout=0):
        super(AttentionalBiGRU, self).__init__()
        self.register_buffer("mask", torch.FloatTensor())

        natt = hid_size * 2

        self.gru = nn.GRU(input_size=inp_size, hidden_size=hid_size, num_layers=1, bias=True, batch_first=True,
                          dropout=dropout, bidirectional=True)
        self.lin = nn.Linear(hid_size * 2, natt)
        self.att_w = nn.Linear(natt, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, packed_batch):

        rnn_sents, _ = self.gru(packed_batch)
        enc_sents, len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        emb_h = self.tanh(self.lin(enc_sents.view(enc_sents.size(0) * enc_sents.size(1), -1)))  # Nwords * Emb

        attend = self.att_w(emb_h).view(enc_sents.size(0), enc_sents.size(1)).transpose(0, 1)
        all_att = self._masked_softmax(attend, self._list_to_bytemask(list(len_s))).transpose(0, 1)  # attW,sent
        attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents

        return attended.sum(0, True).squeeze(0)

    def forward_att(self, packed_batch):

        rnn_sents, _ = self.gru(packed_batch)
        enc_sents, len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)

        emb_h = self.tanh(self.lin(enc_sents.view(enc_sents.size(0) * enc_sents.size(1), -1)))  # Nwords * Emb
        attend = self.att_w(emb_h).view(enc_sents.size(0), enc_sents.size(1)).transpose(0, 1)
        all_att = self._masked_softmax(attend, self._list_to_bytemask(list(len_s))).transpose(0, 1)  # attW,sent
        attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
        return attended.sum(0, True).squeeze(0), all_att

    def _list_to_bytemask(self, l):
        mask = self._buffers['mask'].resize_(len(l), l[0]).fill_(1)

        for i, j in enumerate(l):
            if j != l[0]:
                mask[i, j:l[0]] = 0

        return mask

    def _masked_softmax(self, mat, mask):
        exp = torch.exp(mat) * Variable(mask, requires_grad=False)
        sum_exp = exp.sum(1, True) + 0.0001

        return exp / sum_exp.expand_as(exp)


class HANConnect(nn.Module):
    def __init__(self, ntoken, emb_size=300, hid_size=300):
        super(HANConnect, self).__init__()
        self.emb_size = emb_size  # 嵌入向量大小
        self.ent_dic = None
        self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
        self.word = AttentionalBiGRU(emb_size, hid_size)  # 因为是*2出来的
        self.sent = AttentionalBiGRU(hid_size * 2, hid_size)
        self.sim = F.cosine_similarity
        self.lin_out = nn.Linear(2, 1)  # 线性层  用于输出最后的评分
        self.lin_emb = nn.Linear(hid_size * 2, hid_size)  # 改变维度
        self.register_buffer("docs", torch.Tensor())  # buffer

    def set_emb_tensor(self, emb_tensor):
        self.emb_size = emb_tensor.size(-1)  # # 获取嵌入的维度大小
        self.embed.weight.data = emb_tensor  # # 将传入的tensor设置为嵌入层的权重

    def set_ent_dic(self, ent_dic):  # 设置嵌入向量存储
        self.ent_dic = ent_dic

    def get_sent_view(self, embs, mentions):  # 将mentions的句子表示加入元组中   [([所有备选标号],对应正确标号,对应句子在文档中的标号)]) 调试结束！！解决问题
        mentions = list(mentions)
        for i in range(len(mentions)):
            mentions[i] = [(a, b, embs[i]) for j, (a, b, _) in enumerate(mentions[i])]
        return mentions

    def _reorder_sent(self, sents, stats):  # 重新排序 经调试没有问题
        """

        :param sents: 句子嵌入表示
        :param stats: 统计量
        :return:  docs, lens, real_order, mentions_l
        """
        sort_r = sorted([(l, r, s, m, i) for i, (l, r, s, m) in enumerate(stats)], key=itemgetter(0, 1, 2))
        # 根据统计信息stats对句子进行排序，排序依据是文本长度 文本标号 句子标号 [(文本长度 文本标号 句子标号 [([候选]，真实，句子表示tensor)....])......]
        builder = OrderedDict()  # 有序字典builder，用于根据句子的位置r对句子索引进行分组。
        menbuilder = OrderedDict()  # 用于根据句子位置r对m进行分组
        for (doc_len, doc_id, sent_id, sent_view_mentions, i) in sort_r:
            if doc_id not in builder:
                builder[doc_id] = [i]
                menbuilder[doc_id] = sent_view_mentions
            else:
                builder[doc_id].append(i)
                menbuilder[doc_id].extend(sent_view_mentions)
        # 获得有序字典一个 文本id 和 句子的id 对应 {文本[重新排序的句子id]}
        list_r = list(reversed(builder))  # 反向表示
        docs = self._buffers["docs"].clone().detach().resize_(len(builder), len(builder[list_r[0]]),
                                                              sents.size(1)).fill_(0)
        # docs 的表示 [文档[句子[句子特征]]]
        lens = []
        real_order = []
        mentions_l = []
        for i, x in enumerate(list_r):
            docs[i, 0:len(builder[x]), :] = sents[builder[x], :]
            mentions_l.append(menbuilder[x])
            lens.append(len(builder[x]))
            real_order.append(x)
        real_order = sorted(range(len(real_order)), key=lambda k: real_order[k])
        # docs 的表示 [文档[句子[句子特征]]]
        # lens 的表示 [文档长度]
        # real_order 实际的文本标号顺序列表real_order
        return docs, lens, real_order, mentions_l

    def get_mention_ranks(self, doc_emb, doc_mentions):
        for i in range(len(doc_emb)):  # [([所有备选标号],对应正确标号,句子向量表示)]]
            doc_mentions[i] = [(a, b, c, doc_emb[i]) for _, (a, b, c) in enumerate(doc_mentions[i])]
        # [[([所有备选标号],对应正确标号,句子向量表示,文档向量表示)....].....]
        listall = []
        for doc_men in doc_mentions:
            for mention in doc_men:
                listall.append(mention)
        # [([所有备选标号],对应正确标号,句子向量表示,文档向量表示)....]
        finalInfo = []
        for (candidates, gold_index, sent_view, doc_view) in listall:
            # doc_sim = self.sim(self.lin_emb(doc_view),)
            all_device = doc_view.device  # 确定新建tensor的位置
            scores = torch.Tensor()  # .to(all_device)

            for candidate in candidates:
                if candidate not in self.ent_dic:
                    candidate_emb = F.normalize(torch.randn(self.emb_size, ), dim=0) # 找不到随机初始化
                    self.ent_dic[candidate] = candidate_emb  # 加入词典
                    # print("get!!!!")
                    # candidate_emb = candidate_emb.to(all_device)   # 统一设备
                else:
                    candidate_emb = self.ent_dic[candidate]  # .to(all_device)  # 加载对应特征嵌入
                # print(doc_view.shape[0])
                # print(sent_view.shape[0])
                candidate_emb = candidate_emb.to(all_device)  # 统一设备
                doc_sim = self.sim(candidate_emb, self.lin_emb(doc_view), dim=0).unsqueeze(0).to(all_device)  # 计算相似度
                # print(doc_sim.shape[0])
                sent_sim = self.sim(candidate_emb, self.lin_emb(sent_view), dim=0).unsqueeze(0).to(all_device)
                # print(sent_sim.shape[0])
                # 计算相似度
                temp = torch.cat((doc_sim, sent_sim))
                # print(temp.shape[0])
                # temp = temp.squeeze()
                score = self.lin_out(temp).to(all_device)
                scores = torch.cat((score, scores), dim=0)
            finalInfo.append((scores, gold_index))
        return finalInfo

    def forward(self, batch_sent, stats):
        """
        用于前向传播
        :param batch_sent: 处理好的句子表示 tensor[句子[单词标号]]
        :param stats: 统计数据 排序的(句子长度,文档长度,文档标号,句子在文档中的标号,[([所有备选标号],对应正确标号,对应句子在文档中的标号)])
        :return:
        """
        sent_len, doc_len, doc_id, sent_id, mentions = zip(*stats)
        # 从stats中解压缩
        emb_w = F.dropout(self.embed(batch_sent), training=self.training)
        # 使用self.embed（嵌入层）将batch_reviews中的单词索引转换为词嵌入。
        packed_sents = nn.utils.rnn.pack_padded_sequence(emb_w, torch.Tensor(sent_len), batch_first=True)
        sent_embs = self.word(packed_sents)  # 单词级别注意力结果 14*600
        sent_view_mentions = self.get_sent_view(sent_embs, mentions)  # mentions添加句子表示后的结果
        # [([所有备选标号],对应正确标号,句子向量表示)...]
        doc_embs, lens, real_order, mentions_l = self._reorder_sent(sent_embs,
                                                                    zip(doc_len, doc_id, sent_id,
                                                                        tuple(sent_view_mentions)))
        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(doc_embs, torch.Tensor(lens), batch_first=True)
        doc_embs = self.sent(packed_rev)  # 获取嵌入

        # real_order排序
        final_emb = doc_embs[real_order, :]  # [句子特征]
        final_mentions_l = []
        for i in range(len(real_order)):
            final_mentions_l.append(mentions_l[real_order[i]])  # [([所有备选标号],对应正确标号,句子向量表示)]]
        return self.get_mention_ranks(final_emb, final_mentions_l)
        # 返回所有loss需要的东西
