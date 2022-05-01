import streamlit as st
import torch
from gensim.models import Word2Vec
from gen import gen

model = torch.load("lgg_model_paths/hole-merge_2022-05-01_02_27_10", map_location='cpu')
model.eval()
wv = Word2Vec.load("word_model_paths/hole-merge").wv


st.title("P大树洞-爱的引论")

choice = st.radio("选择一个模型", ("GRU", "LSTM"))

if choice == "GRU":
    with st.spinner("载入模型中..."):
        model = torch.load("lgg_model_paths/hole02_2022-04-01_10_12_06")
        model.eval()
        wv = Word2Vec.load("word_model_paths/hole02").wv

elif choice == "LSTM":
    with st.spinner("载入模型中..."):
        model = torch.load("lgg_model_paths/hole-merge_2022-05-01_02_27_10", map_location='cpu')
        model.eval()
        wv = Word2Vec.load("word_model_paths/hole-merge").wv

count = st.number_input("想要几位小可爱回复你呢？", 1, 20, 5, 1)

dz = st.text_input("发一条树洞吧！")

dz += "\n[Alice]"

with st.spinner("生成中..."):
    out = gen(model, wv, dz, count)

if __name__ == "__main__":
    if out:
        out = out.split("\n")
        for i in out:
            st.write(i)