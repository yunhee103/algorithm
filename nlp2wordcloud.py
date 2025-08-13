# pip install pygame
# pip install pytagcloud
# pip install simplejson

# 동아일보 검색 기능으로 문자열을 읽어 형태소 분석 후 워드클라우드로 출력

from bs4 import BeautifulSoup
import urllib.request 
from urllib.parse import quote

keyword = "무더위"
# keyword = input("검색어 : ")
# print(quote(keyword))

target_url = "https://www.donga.com/news/search?query=" + quote(keyword)
# print(target_url)

source_code = urllib.request.urlopen(target_url)
soup = BeautifulSoup(source_code, 'lxml', from_encoding='utf-8')
# print(soup)

msg = ""
for title in soup.find_all('h4', class_= 'tit'):
    title_link = title.select('a')
    # print(title_link)
    article_url = title_link[0]['href']
    # print(article_url)
    try:
        source_article = urllib.request.urlopen(article_url)
        soup = BeautifulSoup(source_article, 'lxml', from_encoding='utf-8')
        contents = soup.select('div.article_txt')
       #  print(contents)
        for imsi in contents:
            item = str(imsi.find_all(string=True))
            # print(item)
            msg += item

    except Exception as e:
        pass

# print(msg)

from konlpy.tag import Okt
from collections import Counter
okt = Okt()
nouns = okt.nouns(msg)

result = []
for imsi in nouns:
    if len(imsi) > 1:
        result.append(imsi)
# print(result[:10]) # 10개만 출력

count = Counter(result)
# print(count)
tag = count.most_common(50)

import pytagcloud
taglist = pytagcloud.make_tags(tag, maxsize = 100)
# print(taglist)  # [{'color': (201, 107, 174), 'size': 121, 'tag': '재난'},
pytagcloud.create_tag_image(taglist, 'word.png', size=(1000, 600), background=(0,0,0),fontname='korean',rectangular=False)


# 이미지 읽기
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()