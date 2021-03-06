# P大树洞生成器

**中文** | [English](index.md)

## 简介

这是 [***北京大学***](https://www.pku.edu.cn) 国家精品课程**人工智能引论**的课程项目。

本项目受到北京大学[树洞](https://pkuhelper.pku.edu.cn/hole/)的启发。树洞作为匿名交流论坛，是一个天然的自然语言处理语料库（在遵守相关规定的情况下）。

本项目基于因果语言建模（CLM）的原理开发，使用[Python](https://www.python.org)和[PyTorch](https://pytorch.org)实现。

我们在树洞文本数据集上训练LSTM模型，并根据用户的输入，使用训练好的模型产生回复。我们将模型命名为 ***HoleAI***。

您可以在下方链接中亲自尝试不同大小的模型。

## 神经网络架构

- 词嵌入
- 多层LSTM
- 丢弃层
- 层归一化
- 全连接层

## 生成样例
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

## 来当洞主吧！（迫真）

<!-- [😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅😅](https://share.streamlit.io/hirojifukuyama/pkuhole/app.py) -->
[![](assets/hole_icon)](https://share.streamlit.io/hirojifukuyama/pkuhole/app.py)

## 模型细节 @HoleAI

|模型名称|模型大小|隐层大小|LSTM层数|最终验证集损失|
| :------: | :------: | :------: | :------: | :------: |
|HoleAI-small|12.5MB|256|3|0.8276|
|HoleAI-medium|29.2MB|512|2|0.4446|
|HoleAI-large|37.8MB|512|3|0.4366|
|HoleAI-ultra|46.2MB|512|4|0.4714|

## 支持与联系

如果您有任何问题与补充，请发邮件至 **rtzhao1912@gmail.com**。

## 小组成员
- *匡宇轩*，[北京大学信息科学技术学院](https://eecs.pku.edu.cn)
- *陈红韵*，[北京大学信息科学技术学院](https://eecs.pku.edu.cn)
- *王天源*，[北京大学元培学院](https://yuanpei.pku.edu.cn)