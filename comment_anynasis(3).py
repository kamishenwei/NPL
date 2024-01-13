import csv
import heapq
import re
import jieba
import pandas as pd
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import jieba.posseg as pseg

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import confusion_matrix

jieba.setLogLevel(20)  # 设置jieba分词日志级别

# 获取本地停用词表(ntlk自然语言处理库)
stopwords_list = []
index = 0
with open("C:\\Users\\admin\stopwords.txt", 'r', encoding='utf-8') as file:
    stopwords_list = file.readlines()

while index < len(stopwords_list):
    stopwords_list[index] = stopwords_list[index].strip()
    index += 1


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


# 使用示例
file_path = 'C:\\Users\\admin\high_comment.csv'
# csv_data = read_csv_file(file_path)
csv_data = pd.read_csv(file_path)
# 预处理文本数据
# 去除特殊字符和标点符号
csv_data['processed_text'] = csv_data['描述'].str.replace('[^\w\s]', '', regex=False)
# 转换为小写
csv_data['processed_text'] = csv_data['processed_text'].str.lower()
# 转换为str类型
csv_data['processed_text'] = csv_data['processed_text'].astype(str)
# 去除重复值
csv_data.drop_duplicates(subset=['processed_text'], inplace=True)

# 内置程序(好差评对比）
# selected_row = csv_data[(csv_data['类别'] == '内置程序_x0000_') | (csv_data['类别'] == '内置程序')]
# threshold = 0.6
# selected_row['emotion'] = selected_row['sentiments'].apply(lambda x: 1 if x > threshold else 0)
# emotion_counts = selected_row['emotion'].value_counts()
# plt.bar(emotion_counts.index, emotion_counts.values)
# plt.xlabel('Emotion')
# plt.ylabel('Count')
# plt.xticks([0, 1], ['Negative', 'Positive'])
# plt.title('Distribution of Sentiment-inner_exe')
# plt.show()

# 触控
# selected_row = csv_data[(csv_data['类别'] == '触控_x0000_') | (csv_data['类别'] == '触控')]
# threshold = 0.6
# selected_row['emotion'] = selected_row['sentiments'].apply(lambda x: 1 if x > threshold else 0)
# emotion_counts = selected_row['emotion'].value_counts()
# plt.bar(emotion_counts.index, emotion_counts.values)
# plt.xlabel('Emotion')
# plt.ylabel('Count')
# plt.xticks([0, 1], ['Negative', 'Positive'])
# plt.title('Distribution of Sentiment-touch')
# plt.show()

# 通话
# selected_row = csv_data[csv_data['类别'] == '通话']
# threshold = 0.6
# selected_row['emotion'] = selected_row['sentiments'].apply(lambda x: 1 if x > threshold else 0)
# emotion_counts = selected_row['emotion'].value_counts()
# plt.bar(emotion_counts.index, emotion_counts.values)
# plt.xlabel('Emotion')
# plt.ylabel('Count')
# plt.xticks([0, 1], ['Negative', 'Positive'])
# plt.title('Distribution of Sentiment-call')
# plt.show()

# 外观
selected_row = csv_data[(csv_data['类别'] == '外观设计')|(csv_data['类别'] == '外观体验')]
threshold = 0.6
selected_row['emotion'] = selected_row['sentiments'].apply(lambda x: 1 if x > threshold else 0)
emotion_counts = selected_row['emotion'].value_counts()
plt.bar(emotion_counts.index, emotion_counts.values)
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.title('Distribution of Sentiment-looks_and_feelings')
plt.show()

X = csv_data['processed_text']
threshold = 0.6  # 设定阈值
y = [1 if sentiment >= threshold else 0 for sentiment in csv_data['sentiments']]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_val_transformed = vectorizer.transform(X_val)

# model = LogisticRegression()
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# accuracy = model.score(X_val_transformed, y_val)
# print("Accuracy:", accuracy)
y_pred = model.predict(X_val_transformed)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)  # y_val为真实标签，y_pred为预测标签

# 绘制混淆矩阵
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')  # annot=True显示数值，cmap指定颜色图谱
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
ax.yaxis.set_ticklabels(['Negative', 'Positive'])
plt.show()

# 好评可视化
# 对sentiment进行阈值处理
threshold = 0.45
csv_data['emotion'] = csv_data['sentiments'].apply(lambda x: 1 if x > threshold else 0)

# 统计每个情感类别的样本数目
emotion_counts = csv_data['emotion'].value_counts()

# 绘制情感类别的可视化图表
plt.bar(emotion_counts.index, emotion_counts.values)
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.title('Distribution of Sentiment')
plt.show()


def preprocess_text(text):
    # 去除特殊字符和标点符号
    text = re.sub(r"[^\w\s]", "", text)

    # 去除所有英文字符
    text = re.sub(r"[A-Za-z]", "", text)

    # 分词
    words = list(jieba.cut(text))
    words = [word for word in words if word.strip()]  # 去除空格和无意义的词

    # 去除停用词
    stopwords = get_stopwords()  # 获取停用词表，也可以自己定义
    words = [word for word in words if word not in stopwords]

    # 对词性进行标注
    words_with_pos = jieba.lcut(text, True)

    # 标准化文本
    words = [word.lower() for word in words]

    return words


# 统计词频
def get_frequency(text):
    words = preprocess_text(text)
    # 利用Counter统计每个单词的频次
    word_freqs = Counter(words).most_common()
    # top_ten = heapq.nlargest(10, word_freqs.items(), key=lambda x: x[1])
    return word_freqs


# 处理不同类型评论,sublists:经过停词处理后的信息，totallist:处理前相同赛道的评论列表
def get_text(sublists, totallist):
    relist = []
    for col in totallist:
        sublists.append(preprocess_text(col))
    for sublist in sublists:
        relist.extend(sublist)
    tackled_text = ' '.join(relist)

    # 选取对应词性的关键词
    words = pseg.cut(tackled_text)
    text = ''
    for item in words:
        if item.flag == 'a':
            # print("item.word is" + item.word)
            text += item.word + " "

    # print("text------" + text)
    return text


# 获取词频最高的词云图
def get_maxfre_text(sublists, totallist):
    # 初步处理文本信息
    text = get_text(sublists, totallist)
    # print("text-s------" + text)
    result = text.split(" ")
    # print("result------", result)
    freq_dict = {}
    for word in result:
        if word not in freq_dict:
            print(word)
            freq_dict[word] = 0
        freq_dict[word] += 1

    sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    top_ten_words = [item[0] for item in sorted_items]
    text = ' '.join(top_ten_words)
    return text


def get_stopwords():
    # 停用词表，可以根据实际需求进行修改
    stopwords = ["的", "了", "也", "是", "个", "很", "有", "就", "还", "比较", "好", "不错", "很棒", "赞", "值得", "推荐", "吧",
                 "嗯", "啊", "呀", "这样", "那样", "又", "再", "牛", "月", "日", "年", "什么", "那么", "这么", "来", "送", "0000",
                 "哦", "其", "看", "让", "你好", "一点", "发", "真的", "不行", "问", "解决", "微信", "谢谢", "希望", "只能", "发现",
                 "捂 脸", "哈", "有没有", "捂", "脸", "牙", "太", "喜欢", "打开", "情况", "等等", "苹果", "不用", "不好", "特别",
                 "把", "被", "上", "到", "会", "自己", "买", "后", "去", "虽然", "但是", "给", "而且", "并且", "时候", "看看", "呀 ",
                 "吗", "在", "说", "没", "现在", "还是", "都", "就是", "能", "呢", "不", "你们", "今天", "明天", "后天", "昨天",
                 "我们", "不是", "不能", "不可以", "和", "而且", "非常", "正常", "挺", "十分", "不错", "其他", "找到", "弄", "想"
        , "恭喜", "哈", "你", "我", "他", "不愧", "最烦", "笑", "憨", "乐", "快乐", "碎", "憨笑",
                 "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "！", "@", "#", "$", "%", "^",
                 "&", "*", "-", "+", "这个", "那个", "这些", "那些", "这", "那", "要", "一下", "一", "一切", "一定", "一方面", "一次", "一片",
                 "一直", "一个", "可以", "没有", "还有", "哪有", "怎么", "要", "用", "使用", "手机", "魔镜", "里", "店里", "新",
                 "请问", "采用", "呲", "两个", "建议", "服务", "美女", "支持", "自动", "破涕为笑", "抖音", "承诺", "更", "码"]

    # 与ntlk库合并
    stopwords = stopwords + stopwords_list
    return stopwords


# 根据不同特征的评论进行分类
col_index = 4
# 内置程序
inner_comment = [row[col_index] for row in csv_data if row[3] == '内置程序_x0000_']
inner_amount = len(inner_comment)
# 续航赛道
continues_comment = [row[col_index] for row in csv_data if row[3] == '续航赛道_x0000_']
continues_amount = len(continues_comment)
# 性能赛道
qualities_comment = [row[col_index] for row in csv_data if row[3] == '性能赛道_x0000_']
qualities_amount = len(qualities_comment)
# 发热赛道
heat_comment = [row[col_index] for row in csv_data if row[3] == '发热赛道_x0000_']
heat_amount = len(heat_comment)
# 触控
touch_comment = [row[col_index] for row in csv_data if row[3] == '触控_x0000_' or '触控']
touch_amount = len(touch_comment)
# 通话
call_comment = [row[col_index] for row in csv_data if row[3] == '通话_x0000_' or '通话']
call_amount = len(call_comment)
# 无线网络
internet_comment = [row[col_index] for row in csv_data if row[3] == '无线网络_x0000_']
internet_amount = len(internet_comment)
# 音频
video_comment = [row[col_index] for row in csv_data if row[3] == '音频_x0000_']
video_amount = len(video_comment)
# 升级
upgrade_comment = [row[col_index] for row in csv_data if row[3] == '升级_x0000_']
upgrade_amount = len(upgrade_comment)
# 外观
looks_comment = [row[col_index] for row in csv_data if row[3] == '外观设计_x0000_' or '外观设计' or '外观体验_x0000_']
looks_amount = len(looks_comment)

comment = []

# 柱状图显示用户对各项指标的关注程度
label_x = ['Built-in program', 'endurance', 'performance', 'heating', 'touch', 'conversation', 'internet', 'video',
           'upgrade', 'looks']
label_y = [internet_amount, continues_amount, qualities_amount, heat_amount, touch_amount, call_amount, inner_amount,
           video_amount, upgrade_amount, looks_amount]
plt.bar(range(len(label_x)), label_y)
plt.title("Users' attention to various indicators of mobile phones")
plt.xticks([i for i in range(len(label_x))], label_x)
plt.yticks()
for a, b in zip(range(len(label_x)), label_y):
    plt.text(a, b + 0.5, str(b), ha='center')
plt.show()

# 如果要生成完整的词云图，将get_maxfre_text替换成get_text函数
# 生成词云
wordcloud_1 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, qualities_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_1, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_2 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, inner_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_2, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_3 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, continues_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_3, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_4 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, heat_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_4, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_5 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, touch_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_5, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_6 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, call_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_6, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_7 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, internet_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_7, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_8 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, video_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_8, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_9 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, upgrade_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_9, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud_10 = WordCloud(max_words=100, font_path='C:\WINDOWS\Fonts\simhei.ttf', width=1600, height=1200).generate(
    get_maxfre_text(comment, looks_comment))

# 显示词云图
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_10, interpolation='bilinear')
plt.axis('off')
plt.show()

# 统计各种评论的词频
fre = {'内置': [], '续航': [], '性能': [], '发热': [], '触控': [], '通话': [], '网络': [], '音频': [], '升级': [], '外观': []}
fre['内置'].append(get_frequency(get_text(comment, inner_comment)))
fre['续航'].append(get_frequency(get_text(comment, continues_comment)))
fre['性能'].append(get_frequency(get_text(comment, qualities_comment)))
fre['发热'].append(get_frequency(get_text(comment, heat_comment)))
fre['触控'].append(get_frequency(get_text(comment, touch_comment)))
fre['通话'].append(get_frequency(get_text(comment, call_comment)))
fre['网络'].append(get_frequency(get_text(comment, internet_comment)))
fre['音频'].append(get_frequency(get_text(comment, video_comment)))
fre['升级'].append(get_frequency(get_text(comment, upgrade_comment)))
fre['外观'].append(get_frequency(get_text(comment, looks_comment)))

print(fre)
