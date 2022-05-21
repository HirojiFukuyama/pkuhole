# PKU Hole Generator

**English** | [中文](chinese.md)

## Introduction

This is the course project for the NLP class of **Introduction to AI(人工智能引论)** in [***Peking University***](https://www.pku.edu.cn).

The project was mainly inspired by the [hole(树洞)](https://pkuhelper.pku.edu.cn/hole/) in Peking University, which is a perfect corpus for NLP tasks.

It is designed according to the principle of causal language modeling and inplemented in [Python](https://www.python.org) and [PyTorch](https://pytorch.org).

We trained our LSTMs on the hole corpus, and then we used the trained models to generate comments or replies based on the user's input. We name the models ***HoleAI***. 

You can try these models with different sizes on your own with the link provided below.

## Neural Network Architecture

- Word Embedding
- Multi-layer LSTM
- Dropout
- Layer Normalization
- Fully Connected Layer

## Examples
```
😅

[Alice] 表白

[Bob] Bob

[Bob] 来了！

[Carol] 摸摸，因为要留到一个认真的日子！

[Bob] 谢谢A 爱你😘
```
```
popi

[Alice] 身高体重颜值

[洞主] 170，保密，自我感觉中上（会被人偶尔称赞的程度！）

[Bob] Re 洞主: dz是嘉心糖吗

[洞主] Re Bob: 确实

[Alice] Re 洞主: 或者想聊聊也行，我还挺会聊天的
```

## Let's Try!

<!-- [😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅](https://share.streamlit.io/hirojifukuyama/pkuhole/app.py) -->
[![](Unknown)](https://share.streamlit.io/hirojifukuyama/pkuhole/app.py)

## Model Details @HoleAI

|Name|Size|Input words|Hidden size|Number of layers| Final val loss|
| :------: | :------: | :------: | :------: | :------: | :------: |
|HoleAI-small|4.7MB|50|128|3|1.5476|
|HoleAI-medium|12.6MB|50|256|3|0.4562|
|HoleAI-large|37.8MB|30|512|3|0.4354|
|HoleAI-ultra|46.2MB|30|512|4|0.4640|

## Support or Contact

Feel free to email ***rtzhao1912@gmail.com*** if you have any question or supplement.

## Credits
- *Yuxuan Kuang* from [School of EECS, Peking University](https://eecs.pku.edu.cn)
- *Hongyun Chen* from [School of EECS, Peking University](https://eecs.pku.edu.cn)
- *Tianyuan Wang* from [Yuanpei College, Peking University](https://yuanpei.pku.edu.cn)