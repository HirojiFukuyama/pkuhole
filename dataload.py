# some imports
import multiprocessing
import datetime as dt
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_path, input_word_count, vec_size=100, save_word_model=False):
        file = open(data_path, 'r', encoding='utf-8')
        text = file.read()
        file.close()

        word_model = Word2Vec(text, vector_size=vec_size, window=5, min_count=2, 
                            workers=multiprocessing.cpu_count(), epochs=100)

        if save_word_model:
            str = input("请输入即将存储的词汇库的名称：")
            word_model.save("word_model_paths/"+str)

        wv = word_model.wv
        self.vocabulary_length = len(wv)

        # corpus中有的字（词）因为出现次数过少而没有出现在wv中，所以要对他们进行删除
        lst = list(text)
        del_lst = []
        for i in lst:
            if i not in wv.key_to_index:
                del_lst.append(i)
        for i in del_lst:
            lst.remove(i)
        
        self.input_word_count = input_word_count

        self.data = np.array([])

        for i in lst:
            self.data = np.append(self.data, wv.key_to_index[i])

        
    def __len__(self):
        return len(self.data)

        
    def __getitem__(self, idx):
        if idx <= len(self.data)-self.input_word_count:
            return self.data[idx:idx+self.input_word_count]
        else:
            return self.__getitem__(idx-(len(self.data)-self.input_word_count))


# testing and generating word model
if __name__ == '__main__':
    my_dataset = MyDataset('texts/test_dataload.txt', 10, save_word_model=True)
    my_dataloader = DataLoader(my_dataset, batch_size=3, shuffle=True)
    print(next(iter(my_dataloader)).shape)

