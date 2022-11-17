import numpy as np
import pandas as pd
import pickle
import jieba
import os
from tqdm import tqdm

def load_stop_words(file = "stopwords.txt"):
    # 加载停用词
    with open(file,"r",encoding = "utf-8") as f:
        return f.read().split("\n")

def cut_words(file="数学原始数据.csv"):
    stop_words = load_stop_words()
    result = []
    all_data = pd.read_csv(file,encoding = "gbk",names=["data"])["data"]
    for words in all_data:
        c_words = jieba.lcut(words)
        # 注意该式子为列表推导式，其是用
        result.append([word for word in c_words if word not in stop_words])
    return result

def get_dict(data):
    index_2_word = []
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)

    word_2_index = {word:index for index,word in enumerate(index_2_word)}
    word_size = len(word_2_index)

    word_2_onehot = {}
    for word,index in word_2_index.items():
        one_hot = np.zeros((1,word_size))
        one_hot[0,index] = 1
        word_2_onehot[word] = one_hot

    return word_2_index,index_2_word,word_2_onehot

def softmax(x):
    ex = np.exp(x)
    # axis指sum让那个维度消失，x的one hot向量，有2维，keepdims 在这里没有任何区别
    return ex/np.sum(ex,axis = 1,keepdims = True)

if __name__ == "__main__":
    # 对数据集中的每一行数据进行切词
    data = cut_words()
    # 分词结束后获取该语料的3大参数
    word_2_index, index_2_word, word_2_onehot = get_dict(data)


    word_size = len(word_2_index)
    embedding_num = 107
    lr = 0.01
    epoch = 10
    # 相关词范围，设置为左右个3个
    n_gram = 3

    w1 = np.random.normal(-1,1,size = (word_size,embedding_num))
    w2 = np.random.normal(-1,1,size = (embedding_num,word_size))

    # 对每一个epoch循环
    for e in range(epoch):
        # 对语料库中的没有一句话循环
        for words in tqdm(data):
            # 对话中的每一个单词循环
            for n_index, now_word in enumerate(words):
                now_word_onehot = word_2_onehot[now_word]
                # 这里是获取临近的词（注意：切片在最大值上越界其会自动区list边界）
                other_words = words[max(n_index-n_gram,0):n_index] + words[n_index+1 : n_index+1+n_gram]
                for other_word in other_words:
                    other_word_onehot = word_2_onehot[other_word]

                    hidden = now_word_onehot @ w1
                    p = hidden @ w2
                    pre = softmax(p)

                    # 计算loss 优化loss尽量最小，但是在这里不能直接用
                    # loss = -np.sum(other_word_onehot * np.log(pre))

                    # 矩阵求导公式：    d_C = A*d_B
                    # A @ B = C
                    # delta_C = G
                    # delta_A = G @ B.T
                    # delta_B = A.T @ G

                    G2 = pre - other_word_onehot
                    delta_w2 = hidden.T @ G2
                    G1 = G2 @ w2.T
                    delta_w1 = now_word_onehot.T @ G1

                    w1 -= lr * delta_w1
                    w2 -= lr * delta_w2

    with open("word2vec.pkl","wb") as f:
        # 保存训练的参数
        pickle.dump([w1,word_2_index,index_2_word],f)  # word2vec 负采样





