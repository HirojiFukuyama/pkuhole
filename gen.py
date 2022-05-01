import numpy as np
import torch

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def gen(model, wv, dz, count=5, device='cpu'):
    """generate replies"""

    if dz == "\n[Alice]":
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


if __name__ == "__main__":
    pass