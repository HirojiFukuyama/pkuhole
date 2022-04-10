import streamlit as st
import torch
from gensim.models import Word2Vec
from gen import gen

model = torch.load("lgg_model_paths/hole02_2022-04-01_10_12_06")
wv = Word2Vec.load("word_model_paths/hole02").wv

st.title("P大树洞-你的AI小可爱")

count = st.number_input("想要几位小可爱回复你呢？", 1, 10, 5, 1)

dz = st.text_input("发一条树洞吧！")

dz += "\n[Alice]"

with st.spinner("生成中..."):
    out = gen(model, wv, dz, count)

if __name__ == "__main__":
    if out:
        out = out.split("\n")
        for i in out:
            st.write(i)