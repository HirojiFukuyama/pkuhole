# PKU Hole Generator

## Introduction

This is a course project for the NLP class of ***Introduction to AI*** in PKU.

This project aims to simulate the [hole (树洞)](https://pkuhelper.pku.edu.cn/hole/) in PKU. Based on the user's input, the AI models will generate replies that are similar to the real hole replies.

## Environment
- `python==3.9.6`
- `torch==1.11.0`
- `gensim==4.1.2`
- `numpy==1.21.5`
- `tensorboard==2.7.0`
- `streamlit==1.5.0`

<!-- ## File Structure
- Scripts
  - `dataload.py`: data loading and preprocessing
  - `lgg_models.py`: definition of the models
  - `utils.py`: utility functions
  - `main.py`: training on [Boya platform](https://boya.ai.pku.edu.cn/openai/#/index)
  - `train.ipynb`: training locally or on Google Colab
  - `generate.ipynb`: testing generation
  - `testui.py`: testing UI
  - `app.py`: main script for visualization
- Directories 
  - `datasets`: raw txt data
  - `docs`: web material
  - `graph`: pipeline graph
  - `hole`: tensorboard training logs
  - `lgg_model_paths`: saved language models
  - `word_model_paths`: saved word embeddings
  - `checkpoints`: nothing, just ignore it
  - `__pycache__`: system files, just ignore it
- Others
  - `requirements.txt`: required packages for ```app.py```
  - `presentation.pdf`: presentation for the course report
  - `LICENSE`: open source license
  - `readme.md`: this file -->

## How to use
- Approach 1: visit [this page](https://share.streamlit.io/hirojifukuyama/pkuhole/app.py).
- Approach 2: download zip and run ```streamlit run app.py```.

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
- - -
**For more details, check out [our website](https://kryptonite.work/pkuhole) or [our presentation note](presentation.pdf).**