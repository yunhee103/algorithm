# pip install konlpy

from konlpy.tag import Okt, Kkma, Komoran
#corpus(말뭉치) : 자연어 처리를 목적으로 수집된 문자 집단

text = "나는 오늘 아침에 강남에 갔다. 가는 길에 빵집이 보여 너무 먹고 싶었다."

# 형태소 : 의미를 가지는 요소로서는 더 이상 분석할 수 없는 가장 작은 말의 단위

print('Okt --------')
okt = Okt()
print('형태소 : ' , okt.morphs(text))
print('품사 태깅 : ' , okt.pos(text))
print('품사 태깅(어간 포함) : ' , okt.pos(text, stem=True)) #원형 (어근으로 출력)  예) 그래요 -> 그렇다로 출력
print('명사 추출: ' , okt.nouns(text))



print('Kkma --------')
kkma = Kkma()
print('형태소 : ' , kkma.morphs(text))
print('품사 태깅 : ' , kkma.pos(text))
print('품사 태깅(어간 포함) : ' , kkma.pos(text)) 
print('명사 추출: ' , kkma.nouns(text))


print('Komoran --------')
komoran = Komoran()
print('형태소 : ' , komoran.morphs(text))
print('품사 태깅 : ' , komoran.pos(text))
print('품사 태깅(어간 포함) : ' , komoran.pos(text)) 
print('명사 추출: ' , komoran.nouns(text))


print('----워드 클라우드 ----')
# pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
text2 = "나는 오늘 아침에 강남에 갔다. 가는 길에 빵집이 보여 너무 먹고 싶었다. 빵이 특히 강남에 있는"
nouns = okt.nouns(text2)
words = "".join(nouns)
print(words)

wc = WordCloud(font_path='malgun.ttf', width=400, height=300, background_color='white')
cloud = wc.generate(words)

plt.imshow(cloud,interpolation='bilinear')
plt.axis('off')
plt.show()