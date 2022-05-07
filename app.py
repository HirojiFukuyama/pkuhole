import streamlit as st
import torch
from gensim.models import Word2Vec
from gen import gen
import time
import random

@st.cache
def load_wv(wv_path):
    return Word2Vec.load(wv_path).wv

@st.cache
def load_model(model_path):
    return torch.load(model_path, map_location='cpu')

# default:
# last last last one: "lgg_model_paths/hole-merge_lstm_drop_2022-05-02_01_19_40"
# last last one: "lgg_model_paths/hole-merge_input30_2022-05-02_13_24_14"
# last one: "lgg_model_paths/merge_input50"

st.title("P大树洞-爱の引论")
st.subheader("欢迎来到P大树洞！@HoleAI")
choice = st.sidebar.radio("选择一个模型", ("HoleAI-small", "HoleAI-medium", "HoleAI-large", "HoleAI-ultra"))
emotion = st.sidebar.radio("选择一个情绪", ("无", "positive", "negative"))

def preprocess(dz, emotion, positive, negative):
    dz += "\n[Alice]"
    if emotion == 'positive':
        dz += " "
        dz += random.choice(positive)
    elif emotion == 'negative':
        dz += " "
        dz += random.choice(negative)
    return dz

def main():
    flag = False
    count = st.sidebar.number_input("选择树洞长度", 1, 20, 5, 1)
    dz = st.text_input("发一条树洞吧！")
    if dz:
        flag = True

    if st.button("开始生成"):
        wv = load_wv("word_model_paths/hole-merge")

        positive = ['抱抱', 'patpat', '摸摸', 'dz不哭', '呜呜', '哈哈哈哈', '笑死', '嘎嘎', '同感', '哦', '突突突突突突突突', '可爱捏', '恭喜', '哎呀', wv.index_to_key[1713]+'\n', wv.index_to_key[1456]+'\n', wv.index_to_key[1900]+'\n', wv.index_to_key[1111]+'\n', '正确的', '3.92\n', 'dz好棒！', 'www', '哇', '一眼丁真 ', '冲！', '蹲 ']
        negative = ['1/10', wv.index_to_key[273]+'\n', '呵呵', '举报了', '寄\n', '寄了\n', '急了急了', '呃', '典\n', '典中典\n', '麻了', '钝角\n', '怎么会事呢\n', '哈人 ', wv.index_to_key[1343]+'\n', wv.index_to_key[1441]+'\n', wv.index_to_key[2679]+'\n', wv.index_to_key[1566]+'\n', wv.index_to_key[2339]+'\n', '爬\n', '出吗\n']

        dz = preprocess(dz, emotion, positive, negative)

        if choice == "HoleAI-small":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/hole-merge_2022-05-01_04_53_11")
                model.eval()

        elif choice == "HoleAI-medium":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/merge_input50")
                model.eval()

        elif choice == "HoleAI-large":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/merge_input30_512") # 20 epochs
                model.eval()

        elif choice == "HoleAI-ultra":
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/hole-merge_30_512_3_50_2022-05-04_03_17_17") # 50 epochs
                model.eval()

        else:
            with st.spinner("载入模型中..."):
                model = load_model("lgg_model_paths/merge_input50") # medium
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