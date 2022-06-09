import streamlit as st
import torch
from gensim.models import Word2Vec
from utils import *
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def state_cache(func):
    def wrapped(arg):
        cache_id = func.__name__ + str(arg)
        if cache_id not in st.session_state:
            st.session_state[cache_id] = func(arg)
        return st.session_state[cache_id]
    return wrapped


@state_cache
def load_wv(wv_path):
    return Word2Vec.load(wv_path).wv

@state_cache
def load_model(model_path):
    with st.spinner("载入模型中..."):
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model

st.title("P大树洞-爱の引论")
st.subheader("欢迎来到P大树洞！@HoleAI")
choice = st.sidebar.radio("选择一个模型", ("HoleAI-small", "HoleAI-medium", "HoleAI-large", "HoleAI-ultra"), index=2)
emotion = st.sidebar.radio("选择一个情绪", ("neutral", "positive", "negative"))


model_paths = {
    "HoleAI-small": "lgg_model_paths/new/hole-merge_30_256_3_50_2022-06-03_06_55_44",
    "HoleAI-medium": "lgg_model_paths/new/hole-merge_30_512_2_50_2022-06-03_06_57_00",
    "HoleAI-large": "lgg_model_paths/new/hole-merge_30_512_3_50_2022-06-03_03_19_13",
    "HoleAI-ultra": "lgg_model_paths/new/hole-merge_30_512_4_50_2022-06-03_06_54_54"
}

class ProgressBar:
    def __init__(self, total):
        self._pbar = st.progress(0)
        self._total = total
        self._count = 0

    def update(self, count=1):
        self._count += count
        if self._count >= self._total * 0.9:
            self._count = self._total * 0.9
        self._pbar.progress(int(self._count / self._total * 100))

    def finish(self):
        self._pbar.progress(100)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

def main():
    count = st.sidebar.number_input("选择树洞长度", 1, 20, 5, 1)
    dz = st.text_input("发一条树洞吧！")

    if st.button("开始生成") and dz:
        wv = load_wv("word_model_paths/hole_new") # load the word model only once
        wv2 = load_wv("word_model_paths/hole-merge") # original
        dz = preprocess(dz, emotion, wv2)

        model = load_model(model_paths.get(choice, 'large'))

        with ProgressBar(total=count*50) as pbar:
            start_time = time.time()
            st.session_state['out'] = gen(model, wv, dz, count, device=device, step_callback=pbar.update)
            st.balloons()

        if 'out' in st.session_state:
            out = st.session_state['out'] or "没有生成结果"
            for i in out.split("\n"):
                if i != '':
                    st.info(i)
            end_time = time.time()
            st.success("本次生成耗时：{:.4f}秒".format(end_time-start_time))


if __name__ == "__main__":
    main()