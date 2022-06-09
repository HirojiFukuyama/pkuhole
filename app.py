import streamlit as st
import torch
from gensim.models import Word2Vec
from utils import *
import time

st.set_page_config(
    page_title="P大树洞-爱の引论", 
    page_icon="docs/assets/hole_icon",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a course project for the NLP class of ***Introduction to AI*** in PKU.\n\nThis project aims to simulate the [hole (树洞)](https://pkuhelper.pku.edu.cn/hole/) in PKU. Based on the user's input, the AI models will generate replies that are similar to the real hole replies.",
        'Report a bug': "https://github.com/HirojiFukuyama/pkuhole/issues",
        'Get Help': "https://kryptonite.work/pkuhole/"
    }
)

@st.cache
def load_wv(wv_path):
    return Word2Vec.load(wv_path).wv

@st.cache
def load_model(model_path):
    return torch.load(model_path, map_location='cpu')

st.title("P大树洞-爱の引论")
st.subheader("欢迎来到P大树洞！@HoleAI")
choice = st.sidebar.radio("选择一个模型", ("HoleAI-small", "HoleAI-medium", "HoleAI-large", "HoleAI-ultra"), index=2)
emotion = st.sidebar.radio("选择一个情绪", ("neutral", "positive", "negative"))


def main():
    flag = False
    count = st.sidebar.number_input("选择树洞长度", 1, 20, 5, 1)
    dz = st.text_input("发一条树洞吧！")
    if dz:
        flag = True

    if st.button("开始生成"):
        wv = load_wv("word_model_paths/hole_new") # load the word model only once
        wv2 = load_wv("word_model_paths/hole-merge") # original
        dz = preprocess(dz, emotion, wv2)

        if choice == "HoleAI-small":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/new/hole-merge_30_256_3_50_2022-06-03_06_55_44")
                model.eval()

        elif choice == "HoleAI-medium":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/new/hole-merge_30_512_2_50_2022-06-03_06_57_00")
                model.eval()

        elif choice == "HoleAI-large":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/new/hole-merge_30_512_3_50_2022-06-03_03_19_13")
                model.eval()

        elif choice == "HoleAI-ultra":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/new/hole-merge_30_512_4_50_2022-06-03_06_54_54")
                model.eval()

        else:
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/new/hole-merge_30_512_3_50_2022-06-03_03_19_13") # large
                model.eval()

        with st.spinner("生成中..."):
            start_time = time.time()
            out = gen(model, wv, dz, flag, count)

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