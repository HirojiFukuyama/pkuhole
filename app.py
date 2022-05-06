import streamlit as st
import torch
from gensim.models import Word2Vec
from gen import gen
import time

@st.cache
def load_model(model_path, wv_path):
    return torch.load(model_path, map_location='cpu'), Word2Vec.load(wv_path).wv

# default:
# last last last one: "lgg_model_paths/hole-merge_lstm_drop_2022-05-02_01_19_40"
# last last one: "lgg_model_paths/hole-merge_input30_2022-05-02_13_24_14"
# last one: "lgg_model_paths/merge_input50"

st.title("P大树洞-爱の引论")
st.subheader("欢迎来到P大树洞！@HoleAI")
choice = st.radio("选择一个模型", ("HoleAI-small", "HoleAI-medium", "HoleAI-large", "HoleAI-ultra"))

def main():
    count = st.number_input("选择树洞长度", 1, 20, 5, 1)
    dz = st.text_input("发一条树洞吧！")
    dz += "\n[Alice]"
    
    if st.button("开始生成"):
        if choice == "HoleAI-small":
            with st.spinner("载入模型中..."):
                model, wv = load_model("lgg_model_paths/hole-merge_2022-05-01_04_53_11", "word_model_paths/hole-merge")
                model.eval()

        elif choice == "HoleAI-medium":
            with st.spinner("载入模型中..."):
                model, wv = load_model("lgg_model_paths/merge_input50", "word_model_paths/hole-merge")
                model.eval()

        elif choice == "HoleAI-large":
            with st.spinner("载入模型中..."):
                model, wv = load_model("lgg_model_paths/merge_input30_512", "word_model_paths/hole-merge") # 20 epochs
                model.eval()

        elif choice == "HoleAI-ultra":
            with st.spinner("载入模型中..."):
                model, wv = load_model("lgg_model_paths/hole-merge_30_512_3_50_2022-05-04_03_17_17", "word_model_paths/hole-merge") # 50 epochs
                model.eval()

        else:
            with st.spinner("载入模型中..."):
                model, wv = load_model("lgg_model_paths/merge_input50", "word_model_paths/hole-merge")
                model.eval()

        with st.spinner("生成中..."):
            start_time = time.time()
            out = gen(model, wv, dz, count)

        if out:
            out = out.split("\n")
            for i in out:
                if i != '':
                    st.info(i)
            end_time = time.time()
            st.success("本次生成耗时：{:.4f}秒".format(end_time-start_time))
            st.balloons()

if __name__ == "__main__":
    main()