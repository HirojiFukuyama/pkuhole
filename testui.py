import streamlit as st
import torch
from gensim.models import Word2Vec
from gen import gen
import time
# st.title("Hello World!")


model = torch.load("lgg_model_paths/hole-merge_2022-05-01_04_53_11", map_location='cpu') # input50
model.eval()
wv = Word2Vec.load("word_model_paths/hole-merge").wv

# count = 2
# dz = "我是一个树洞"
# out = gen(model, wv, dz, count)
# if out:
#     out = out.split("\n")
#     print(out)

print(model)