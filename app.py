import streamlit as st
import torch
from gensim.models import Word2Vec
from gen import gen


# last last one: "lgg_model_paths/hole-merge_lstm_drop_2022-05-02_01_19_40"
# last one: "lgg_model_paths/hole-merge_input30_2022-05-02_13_24_14"
model = torch.load("lgg_model_paths/Epoch_20", map_location='cpu') # input50
model.eval()
wv = Word2Vec.load("word_model_paths/hole-merge").wv


st.title("P大树洞-爱の引论")

choice = st.radio("选择一个模型", ("LSTM", "GRU"))

if choice == "GRU":
    with st.spinner("载入模型中..."):
        model = torch.load("lgg_model_paths/hole-merge_2022-05-01_04_53_11", map_location='cpu')
        model.eval()
        wv = Word2Vec.load("word_model_paths/hole-merge").wv

elif choice == "LSTM":
    with st.spinner("载入模型中..."):
        model = torch.load("lgg_model_paths/Epoch_20", map_location='cpu')
        model.eval()
        wv = Word2Vec.load("word_model_paths/hole-merge").wv

count = st.number_input("选择回复数量", 1, 20, 5, 1)

dz = st.text_input("发一条树洞吧！")

dz += "\n[Alice]"

with st.spinner("生成中..."):
    out = gen(model, wv, dz, count)

if __name__ == "__main__":
    if out:
        out = out.split("\n")
        for i in out:
            st.write(i)