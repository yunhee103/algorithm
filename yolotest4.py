# 탐지된 객체에 대한 설명을 글로 달아주기
# OpenCV와 YOLO 모델을 사용해 이미지에서 객체를 감지하고, 감지된 객체에 대한 정보를 출력 및 파일로 기록하는 프로그램
import urllib.parse
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# 객체 설명과 링크 제공
object_info = {
    "person": {
        "description": "이 객체는 사람이 감지된 경우입니다. 사람 감지는 보안 감시, 출입 관리 시스템 등에 매우 유용합니다. 또한 얼굴 인식, 행동 분석 등 다양한 분야에 적용됩니다.",
        "use_case": "사람 감지는 보안 시스템에서 출입 관리, 비상 상황에서의 대처, 헬스케어 분야에서 노인 및 환자의 상태 모니터링에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("사람"))
    },
    "car": {
        "description": "이 객체는 자동차가 감지된 경우입니다. 자동차 감지는 교통 흐름 분석, 불법 주차 감시, 사고 예방 등 다양한 분야에 활용됩니다.",
        "use_case": "자동차 감지는 자율 주행 시스템, 스마트 교통 시스템, 교차로 모니터링 등에 활용되며, 도시 계획 및 교통 관리에도 중요한 역할을 합니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("자동차"))
    },
    "truck": {
        "description": "이 객체는 트럭이 감지된 경우입니다. 트럭 감지는 물류 창고 관리, 도로 교통 모니터링, 고속도로에서의 추적 등에 활용됩니다.",
        "use_case": "트럭 감지는 물류 효율화, 고속도로 사고 예방, 교통량 분석 등에 사용되며, 스마트 물류 및 재난 관리 시스템에도 중요합니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("트럭"))
    },
    "motorcycle": {
        "description": "이 객체는 오토바이가 감지된 경우입니다. 오토바이 감지는 교통 사고 예방 시스템, 도로에서의 차량 추적 등에 사용됩니다.",
        "use_case": "오토바이 감지는 도로 교통 사고 예방, 긴급 상황 대응, 스마트 교통 시스템 등에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("오토바이"))
    },
    "dog": {
        "description": "이 객체는 강아지가 감지된 경우입니다. 강아지 감지는 반려동물 보호, 유기 동물 탐지 및 동물원 관리 등에서 중요합니다.",
        "use_case": "강아지 감지는 동물 보호 시스템, 유기 동물 탐지 시스템 및 스마트 펫 모니터링 시스템에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("강아지"))
    },
    "cat": {
        "description": "이 객체는 고양이가 감지된 경우입니다. 고양이 감지는 스마트 펫 모니터링 시스템과 연계되어 유용하게 사용됩니다.",
        "use_case": "고양이 감지는 반려동물 모니터링 시스템, 동물원 관리 및 스마트 홈 시스템에 활용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("고양이"))
    },
    "bus": {
        "description": "이 객체는 버스가 감지된 경우입니다. 버스 감지는 대중교통 분석, 버스 전용차로 감시 및 혼잡도 모니터링 등에 활용됩니다.",
        "use_case": "버스 감지는 스마트 시티 교통 시스템, 버스 정류장 혼잡도 분석 및 통근 시간 최적화에 사용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("버스"))
    },
    "bird": {
        "description": "이 객체는 새가 감지된 경우입니다. 새 감지는 자연 생태 모니터링, 조류 충돌 방지 시스템 등에 활용됩니다.",
        "use_case": "새 감지는 공항의 조류 충돌 방지 시스템, 야생 동물 보호 구역의 생태계 분석, 스마트 환경 감시 시스템에 활용됩니다.",
        "link": "https://ko.wikipedia.org/wiki/{}".format(urllib.parse.quote("새"))
    }
}

model = YOLO('yolov8m.pt')  # 'yolov8m.pt' 모델을 로드함. 'm'은 'medium' 버전을 의미함.
image_path = 'yotest2.jpg'
image = cv2.imread(image_path)
if image is None:           # 이미지를 제대로 읽어왔는지 확인하는 조건문.
    print('이미지 읽기 실패')
    exit()

results = model(image) # 이미지에 대해 YOLO 모델을 실행하여 객체를 감지함.
detected_obj = [] # 감지된 객체 라벨들을 저장할 리스트를 만듦.

for result in results:   # 감지 결과 리스트를 반복함.
    for box in result.boxes: # 각 결과의 바운딩 박스 정보를 반복함.
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # 바운딩 박스 좌표를 정수형으로 추출함.
        label = result.names[int(box.cls[0])] # 감지된 객체의 라벨을 얻음.
        confidence = box.conf[0].item() # 감지된 객체의 신뢰도를 얻음.
        detected_obj.append(label) # 감지된 라벨을 리스트에 추가함.

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2) # 이미지에 바운딩 박스를 그림.
        cv2.putText(image,f'{label} {confidence:.2f}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # 바운딩 박스 위에 라벨과 신뢰도를 텍스트로 표시함.

# print('detected obj :' , detected_obj)  # detected obj : ['person', 'person', 'motorcycle', 'person', 'motorcycle', 'person', 'car', 'car', 'car']

# 결과 이미지 저장 (시간 별 저장)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')   # 현재 시간을 '년월일_시분초' 형식의 문자열로 만듦.
# print('timestamp : ', timestamp)


output_path = f'yolotest4_{timestamp}.jpg' # 저장할 이미지 파일의 경로를 만듦.
cv2.imwrite(output_path, image) # 바운딩 박스가 그려진 이미지를 저장함.
print(f'탐지된 객체가 {output_path}로 저장') # 저장 완료 메시지를 출력함.

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # BGR 이미지를 RGB로 변환하여 Matplotlib로 표시함.
plt.axis('off') # 그래프 축을 숨김.
plt.show() # 이미지를 화면에 보여줌.



# 감지된 이미지에 설명 및 링크 출력
description_text = '' # 출력할 설명을 담을 빈 문자열을 만듦.

for obj in set(detected_obj): # 감지된 객체 리스트에서 중복을 제거하고(set), 각 객체에 대해 반복함.
    if obj in object_info: # 현재 객체가 `object_info` 딕셔너리에 있는지 확인함.
        description_text += f'\n{obj} 탐지됨 :\n' # 객체 라벨과 함께 줄바꿈을 추가함.
        description_text += f'설명 : {object_info[obj]["description"]}\n' # 딕셔너리에서 해당 객체의 설명을 가져와 추가함.
        description_text += f'사용사례 : {object_info[obj]["use_case"]} \n' # 사용 사례를 가져와 추가함.
        description_text += f'자세한 내용 : {object_info[obj]["link"]} \n' # 위키백과 링크를 가져와 추가함.
        
print('\n 객체 설명 :', description_text) # 최종 완성된 설명 텍스트를 출력함.

# 감지 결과 로그 파일 저장
log_file = 'yolotest4log.txt' # 로그 파일명을 지정함.
with open(log_file, 'a', encoding='utf-8') as log: # 로그 파일을 '추가(a)' 모드로 염. 파일이 없으면 새로 만듦.
    log.write(f"[{timestamp}] 감지된 객체 :{','.join(set(detected_obj))}\n") # 로그에 현재 시간과 감지된 객체 목록을 씀.
    log.write(description_text + '\n\n') # 객체 설명 텍스트를 로그에 씀.

print(f'{log_file}에 저장됨') # 로그 파일에 저장 완료 메시지를 출력함.

"""
1. 객체 감지 모델의 버전
코드에서 YOLO('yolov8n.pt')와 YOLO('yolov8m.pt') 두 가지 모델을 사용하고 있는데, 이 둘의 차이를 이해

yolov8n (nano): 모델 크기가 가장 작아서 속도가 매우 빠릅니다. 실시간으로 웹캠 영상을 처리할 때 적합하지만, 정확도는 조금 낮을 수 있음.

yolov8m (medium): yolov8n보다 크기가 크고 속도는 약간 느리지만, 감지 정확도가 더 높음. 단일 이미지를 분석하거나 더 높은 정확도가 필요할 때 유리.


2. 로그 파일 관리
마지막 코드에서 감지 결과를 yolotest4log.txt 파일에 기록하는 기능이 추가.

with open(log_file, 'a', encoding='utf-8') as log:: 여기서 **'a'**는 파일을 'append' 모드로 연다는 의미. 
기존 파일 내용에 새로운 내용을 덧붙여 저장하기 때문에, 코드를 여러 번 실행해도 이전 로그가 사라지지 않음. 
만약 'w' (write) 모드를 사용하면, 코드를 실행할 때마다 파일 내용이 덮어쓰여서 이전 기록이 모두 지워짐.

','.join(set(detected_obj)): set() 함수는 리스트(detected_obj)에서 중복된 항목을 제거 해줌. 만약 한 이미지에 'person'이 여러 명 감지되었다면, set()을 통해 중복을 없애고 'person'이라고 한 번만 기록할 수 있음. 
','.join()은 중복이 제거된 객체들을 쉼표(,)로 구분된 하나의 문자열로 만들어 줌.

3. URL 인코딩
urllib.parse.quote("사람") 코드가 사용되었는데, 이는 URL 인코딩을 위한 것.

URL 인코딩: URL 주소에는 한글이나 특수 문자가 포함될 수 없음. 
urllib.parse.quote() 함수는 한글과 같은 문자를 %EC%82%AC%EB%9E%8C와 같이 웹에서 인식할 수 있는 형태로 변환해 줌. 
따라서 한국어 위키백과 링크를 올바르게 생성할 수 있게 해주는 중요한 역할을 함.
"""