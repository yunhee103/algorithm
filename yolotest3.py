# 단일 이미지 파일(image1.jpg)을 처리하는 코드
import cv2 # OpenCV 라이브러리를 가져옴. 이미지 처리에 사용됨.
from ultralytics import YOLO # Ultralytics에서 YOLO 모델을 가져옴. 객체 감지에 사용됨.
import numpy as np # NumPy 라이브러리를 가져옴. 배열 연산에 사용됨.
import matplotlib.pyplot as plt # Matplotlib 라이브러리를 가져옴. 이미지 시각화에 사용됨.
import time # 시간 관련 함수를 다루기 위해 가져옴.
import os # 파일 시스템 관련 작업을 위해 가져옴.
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # plt 나왔다가 사라지면 걸기

model = YOLO('yolov8n.pt')

image_path = 'image1.jpg' # 처리할 이미지 파일 경로를 'image1.jpg'로 지정함.


try:
    image = cv2.imread(image_path)
except FileNotFoundError as e:
    print('에러 : ', e)
    raise SystemExit

original = image.copy() # 원본 이미지 데이터를 백업함. 추후 바운딩 박스 없이 객체를 잘라낼 때 사용됨.
results = model(image)  # YOLO 모델을 이미지에 적용해 객체 감지를 수행함.
# print(results)


person_count = 0     # 'person' 객체의 수를 세기 위한 변수임.
for result in results:  # 감지된 결과 리스트를 반복함.
    for box in result.boxes:     # 각 결과의 바운딩 박스 정보를 반복함.
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표를 정수형으로 추출함.
        label = result.names[int(box.cls[0])] # 감지된 객체의 라벨(이름)을 얻음.
        confidence = box.conf[0].item() # 감지된 객체의 신뢰도(확률)를 얻음.

        if label == 'person':   # 감지된 객체가 'person'인지 확인하는 조건문임.
            person_count += 1   # 'person' 객체 수를 1 증가시킴.
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2) # 이미지에 바운딩 박스를 그림.
        cv2.putText(image,f'{label} {confidence:.2f}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # 바운딩 박스 위에 라벨과 신뢰도를 텍스트로 표시함.

print(f'감지된 사람 수 : {person_count}') # 감지된 총 사람 수를 출력함.
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # OpenCV(BGR) 이미지를 Matplotlib(RGB) 형식으로 변환하여 화면에 표시함.
plt.axis('off') # 그래프 축을 숨김.
plt.title(f'detected person : {person_count}people') # 그래프 제목을 설정함.
plt.show() # 이미지를 창에 띄워줌.


# 바운딩 박스된 이미지 전체를 저장
out_path = 'outtest3.jpg'
cv2.imwrite(out_path, image)
print('바운딩 박스된 이미지 저장 완료')

# 바운딩 박스 내부 객체만 저장
for idx, result in enumerate(results):  # 결과를 순서(idx)와 함께 반복함.
    for j, box in enumerate(result.boxes): # 각 박스 정보를 순서(j)와 함께 반복함.
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # 바운딩 박스 좌표를 얻음.
        label = result.names[int(box.cls[0])] # 라벨을 얻음.
        confidence = box.conf[0].item() # 신뢰도를 얻음.


        # 원본 이미지에서 ROI(region of interest, 관심영역) 추출
        cropped = image[y1:y2, x1:x2]   # 바운딩 박스가 그려진 이미지에서 객체 영역을 잘라냄. 따라서 잘라낸 이미지에 바운딩 박스 선이 포함됨. # 행 방향 ,열 방향 image(H, W, 3) 배열 슬라이싱을 통해 선택된 이미지 배열 반환
        print('cropped :', cropped)

        # 선택된 이미지 배열 저장
        crop_path = f'crop_{idx}_{j}_{label}_{confidence:.2f}.jpg'   # 잘라낸 객체 이미지의 파일명을 만듦.
        cv2.imwrite(crop_path, cropped) # 잘라낸 이미지를 저장함.
        print(f'객체 {label}이 저장됨') # 저장 완료 메시지를 출력함.


# 바운딩 박스 내부 객체만 저장 (박스 선 없이 저장)
for idx, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        # 원본 이미지에서 ROI(region of interest, 관심영역) 추출
        cropped = original[y1:y2, x1:x2]   # **원본 이미지**에서 객체 영역을 잘라냄. 이 때문에 바운딩 박스 선이 포함되지 않음.
        print('cropped :', cropped)

        # 선택된 이미지 배열 저장
        crop_path = os.path.join('crops', f'crop_{idx}_{j}_{label}_{confidence:.2f}.jpg')    # 'crops' 폴더에 저장할 경로를 만듦.
        cv2.imwrite(crop_path, cropped)
        print(f'객체 {label}이 저장됨: {crop_path}')

# 감지된 객체의 중심 좌표 출력

p_count = 0      # 'person' 객체의 수를 세기 위한 변수임.

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        center_x = (x1 + x2) // 2 # 바운딩 박스의 중심 x좌표를 계산함.
        center_y = (y1 + y2) // 2 # 바운딩 박스의 중심 y좌표를 계산함.

        if label.lower() == 'person':   # 라벨이 'person'인지 확인하는 조건문임.
            p_count += 1 # 사람 수를 증가시킴.
            print(f'person => {p_count}: 중심좌표는 {center_x},{center_y}, 신뢰도 :{confidence:.2f}') # 사람의 중심 좌표와 신뢰도를 출력함.
            # 중심적 그리기
            cv2.circle(image, (center_x, center_y), 5, (0,0,255), -1) # 이미지에 빨간색 원으로 중심점을 표시함.
            coord_text = f'({center_x}, {center_y})' # 중심 좌표 텍스트를 만듦.
            cv2.putText(image, coord_text, (center_x+10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2) # 이미지에 중심 좌표 텍스트를 표시함.
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2) # 모든 객체에 초록색 바운딩 박스를 그림.
        cv2.putText(image,f'{label} {confidence:.2f}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # 모든 객체에 라벨과 신뢰도를 표시함.



plt.figure(figsize=(10,8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


"""
바운딩 박스(Bounding Box) 추출 방식과 색상 공간 변환에 대한 부분입니다.

바운딩 박스 추출 방식의 차이점 
코드에는 바운딩 박스 내부 객체를 저장하는 부분이 두 개 있음. 다른점 확인하기 (테두리 유무).

첫 번째 저장 루프:

cropped = image[y1:y2, x1:x2]
이 부분은 바운딩 박스가 그려진 이미지(image)에서 객체 영역을 잘라냄. 따라서 잘려진 이미지에는 바운딩 박스를 그리는 초록색 선이 포함.

두 번째 저장 루프:

cropped = original[y1:y2, x1:x2]
이 부분은 **바운딩 박스를 그리기 전 원본 이미지(original)**에서 객체 영역을 잘라냄. 이 방법 덕분에 잘려진 이미지에는 바운딩 박스 선이 포함되지 않고, 순수한 객체 이미지만 남게 됨.

이 두 가지 방식은 필요에 따라 적절하게 사용될 수 있습니다. 일반적으로는 original 변수를 사용하는 두 번째 방식이 객체만 추출하기 때문에 더 유용.

BGR과 RGB 색상 공간
OpenCV와 Matplotlib는 이미지를 다루는 방식에 차이가 있어, 이 둘을 함께 사용할 때 색상 공간을 변환해야 함.

OpenCV: 기본적으로 BGR(파랑-초록-빨강) 순서로 색상 채널을 처리. cv2.imread() 함수도 BGR 순서로 이미지를 읽음.

Matplotlib: 기본적으로 RGB(빨강-초록-파랑) 순서로 색상 채널을 처리.

만약 변환 없이 OpenCV로 읽은 이미지를 Matplotlib로 바로 보여주면, 파란색과 빨간색이 서로 바뀌어 부자연스러운 색상으로 나타남. 
코드에서 사용된 cv2.cvtColor(image, cv2.COLOR_BGR2RGB)는 이러한 문제점을 해결하기 위해 BGR 이미지를 RGB로 변환해주는 역할.
"""