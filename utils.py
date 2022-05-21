import numpy as np
import torch
import random

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def gen(model, wv, dz, flag, count=5, device='cpu'):
    """generate replies"""

    if flag == False:
        return

    del_lst = []
    lst = list(dz)

    for i in lst:
        if i not in wv.key_to_index:
            del_lst.append(i)
    for i in del_lst:
        lst.remove(i)

    data = np.array([])
    for i in lst:
        data = np.append(data, wv.key_to_index[i])

    i = 0

    while i < count:
        data = np.stack((data,))
        x = torch.Tensor(data)
        x = x.to(torch.long).to(device)
        y = model(x)[0][-1]
        p = y.detach().numpy()
        p = softmax(p)

        idx = np.random.choice(np.arange(len(wv)), p=p)
        new_word = wv.index_to_key[idx]

        lst.append(new_word)
        data = np.append(data, idx)

        if new_word == '\n':
            i += 1
    
    out = "".join(lst)

    return out


def preprocess(dz, emotion, wv):
    # "knowledge"
    positive = ['抱抱', 'patpat', '摸摸', 'dz不哭', '呜呜', '哈哈哈哈', '笑死', '嘎嘎', '同感', '哦', '突突突突突突突突', '可爱捏', '恭喜', '哎呀', wv.index_to_key[1713]+'\n', wv.index_to_key[1456]+'\n', wv.index_to_key[1900]+'\n', wv.index_to_key[1111]+'\n', '正确的', '3.92\n', 'dz好棒！', 'www', '哇', '一眼丁真 ', '冲！', '蹲 ', '难蚌\n', '蚌埠住了\n', '正确的 ']
    negative = ['1/10', wv.index_to_key[273]+'\n', '呵呵', '举报了', '寄\n', '寄了\n', '急了急了', '呃', '典\n', '典中典\n', '麻了', '钝角\n', '怎么会事呢\n', '哈人 ', wv.index_to_key[1343]+'\n', wv.index_to_key[1441]+'\n', wv.index_to_key[2679]+'\n', wv.index_to_key[1566]+'\n', wv.index_to_key[2339]+'\n', '爬\n', '出吗\n', "哼哼哼啊啊啊啊啊啊啊啊啊啊啊啊啊啊\n"]

    dz += "\n[Alice]"
    if emotion == 'positive':
        dz += " "
        dz += random.choice(positive)
    elif emotion == 'negative':
        dz += " "
        dz += random.choice(negative)
    return dz


if __name__ == "__main__":
    print("utils.py")