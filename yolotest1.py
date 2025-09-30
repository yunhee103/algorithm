# pip install opencv-python, ultralytics

# from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # plt 나왔다가 사라지면 걸기
 # Matplotlib를 사용할 때 발생하는 오류를 방지하기 위한 설정. 특히 plt.show()가 창을 제대로 띄우지 못하고 바로 종료되는 문제를 해결.
import sys
import subprocess
import sys
import subprocess


# try:
#     model = YOLO('yolov8n.pt')

# except Exception as e:
#     print('처리 오류 : ', e)

#     sys.exit()


# print(model.names)  #COCO dataset class80개
# print(len(model.names))

# 이 부분은 'ultralytics' 라이브러리가 설치되어 있지 않으면 자동으로 pip 명령어를 사용해 설치를 시도하는 코드 블록

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print('ultralytics가 설치되지 않아 설치를 시작합니다.')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    except subprocess.CalledProcessError as e:
        raise SystemExit(f'ultralytics 설치 실패. 수동으로 설치하세요')
    from ultralytics import YOLO

import ultralytics
ultralytics.checks()     # ultralytics 라이브러리가 올바르게 설치되었는지 확인하는 함수를 호출

try:
    model=YOLO('yolov8n.pt')     # 'yolov8n.pt'라는 미리 학습된 YOLOv8 모델을 로드. 'n'은 nano 버전을 의미하며, 빠르고 가벼움
except Exception as e:
    print(f'error loading model: {e}')

    sys.exit()


print(model.names)  # YOLOv8 모델이 감지할 수 있는 객체들의 이름(클래스)을 출력 #COCO dataset class80개
print(len(model.names)) # 감지 가능한 클래스(객체)의 총 개수를 출력

# 이미지 로딩 후 객체 감지 연습
from PIL import Image       # Pillow 라이브러리에서 Image 모듈. 이미지 파일을 다루는 데 사용됨
import cv2      #컴퓨터 비전, 영상처리, 머신러닝 영상관련 기능 제공 
import numpy as np
import matplotlib.pyplot as plt

image_path = 'dog.jpg'  # 감지할 이미지 파일의 경로를 지정

try:
    image = Image.open(image_path)   # Pillow를 사용해 지정된 경로의 이미지를 열음
    plt.imshow(image)
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f'error loading image: {e}')
    exit()

try:
    results = model(image)  # 로드된 이미지에 대해 YOLO 모델을 실행하여 객체를 감지

except Exception as e:
    print(f'err during inference: {e}')
    exit()

# print(results)
print(results[0].orig_shape)    # 원본크기 # 감지 결과에서 원본 이미지의 크기(높이, 너비)를 출력.

# pillow -> numpy 배열로 변환
image = np.array(image)
print(image.shape)  # 변환된 NumPy 배열의 형태(shape), 즉 (높이, 너비, 채널 수)를 출력
print(image[:5,:5]) # 이미지 배열의 좌측 상단 5x5 픽셀 값을 출력
print(image[0,0])   # 이미지 배열의 가장 좌측 상단 픽셀의 값을 출력

cropped = image[:100, :100] # 원본 이미지 배열에서 좌측 상단 100x100 픽셀 영역을 잘라냄
print('cropped :', cropped)
plt.imshow(cropped) # 잘라낸 이미지를 표시
plt.axis('off')
plt.show()

# 감지된 객체 이미지에 박스 채우기
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # Matplotlib(RGB)에서 OpenCV(BGR)로 색상 채널 순서를 변환. 이는 OpenCV 함수들이 BGR 형식을 사용하기 때문.

for result in results:
    try:
        for box in result.boxes:        # 바운딩박스 리스트 정보
            x1, y1,x2, y2 = map(int,box.xyxy[0])        # [0]텐서  -> int로 변환
            print(x1,y1,x2,y2)  # 11 20 133 151
            label = result.names[int(box.cls[0])]       # cls : 클래스 /  바운딩 박스의 클래스 ID를 사용해 객체 이름을 얻음
            print(label)        # result.names의 16번째 value가 dog
            confidence = box.conf[0].item()  # float으로 받을 수 있음 / 신뢰도(확률) 얻기
            print('confidence :', confidence)   # confidence : 0.46209466457366943

            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)    # 이미지에 초록색 바운딩 박스를 그림. (0,255,0)은 BGR 색상으로 초록색.
            cv2.putText(image, f'{label} {confidence:.2f}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)    # 바운딩 박스 위에 라벨과 신뢰도를 텍스트로 표시

    except Exception as e:
        print(f'err : processing:{e}')

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV(BGR) 이미지를 다시 Matplotlib(RGB) 형식으로 변환하여 최종 결과를 표시
plt.axis('off')
plt.show()

cv2.imwrite('outtest1.jpg', image)  # 바운딩 박스가 그려진 최종 이미지를 'outtest1.jpg' 파일로 저장


"""
- 딥러닝과 컴퓨터 비전의 기초

모델 추론(Inference): 모델 학습이 끝난 후, 새로운 데이터를 입력받아 결과를 예측하는 과정을 '추론(Inference)'이라고 힘. 
results = model(image) 코드가 바로 이 추론을 수행하는 부분. 훈련된 모델을 실제 서비스에 적용하는 단계.

전이 학습(Transfer Learning): YOLO 모델처럼 대규모 데이터셋(예: COCO)으로 미리 학습된 모델을 가져와서(pre-trained model), 새로운 데이터셋에 맞게 미세 조정하는 기술. 처음부터 모델을 학습시키는 것보다 훨씬 효율적.

텐서(Tensor): 딥러닝 모델에서 사용하는 다차원 배열을 의미합니다. NumPy 배열과 유사하지만, GPU 연산에 최적화되어 있습니다. 코드에서 box.xyxy[0]처럼 결과값을 다룰 때 텐서 형태로 반환.

- 코드 심화 및 최적화
cv2.CAP_DSHOW: 윈도우 환경에서 카메라를 열 때 발생하는 문제를 해결하기 위한 백엔드(backend) 설정. 웹캠 연결이 불안정할 때 사용하면 좋음.

ord('q'): waitKey() 함수는 키보드 입력을 아스키코드 값으로 반환. 
ord() 함수는 문자를 아스키코드로 변환해 주기 때문에, ord('q')는 'q' 키를 눌렀을 때의 아스키코드 값과 비교하여 종료 조건을 만듬.

confidence 임계값 설정: 현재 코드는 모든 감지 결과를 처리하지만, 신뢰도가 낮은 객체는 무시하도록 설정할 수 있음. 
예를 들어, if confidence > 0.5:와 같은 조건을 추가하여 정확도가 높은 결과만 사용하도록 코드를 개선할 수 있음.

- 확장 가능한 프로젝트 아이디어
실시간 객체 추적(Object Tracking): 단순 감지를 넘어, 특정 객체가 영상 내에서 어떻게 움직이는지 추적하는 기능.
OpenCV의 Tracker 모듈이나 DeepSORT와 같은 별도 라이브러리를 사용해 구현.

객체 중심점 기반 거리 측정: 카메라 시야에 들어온 객체의 중심점을 기준으로 화면의 특정 영역과의 거리를 계산하여 객체가 얼마나 가까이 또는 멀리 있는지 추정할 수 있음.

사용자 인터페이스(UI) 개발: Tkinter, PyQt5, 또는 Streamlit 같은 라이브러리를 사용하여 감지된 결과를 보여주는 GUI를 만들 수 있음. 실시간 영상과 함께 감지된 객체 수, 신뢰도 등을 시각적으로 표시
"""