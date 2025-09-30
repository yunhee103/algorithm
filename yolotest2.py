"""import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0번 카메라(기본 웹캠)

if not cap.isOpened():
    print("웹캠을 열 수 없네 ㅠㅠ.")
else:
    print("웹캠이 열렸습니다. ESC 눌러 종료하세요.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()"""

import cv2  # OpenCV 라이브러리. 이미지 및 비디오 처리에 사용됨.
from ultralytics import YOLO     # Ultralytics에서 YOLO 모델을 가져옴. 객체 감지에 사용됨.
import time # 시간 관련 함수
import os   # 파일 시스템 관련 작업

model = YOLO('yolov8n.pt')   # 미리 학습된 YOLOv8n 모델을 로드함. 'n'은 'nano' 버전을 의미하며, 빠르고 가벼움.
print(model.names)           # 모델이 감지할 수 있는 80개 객체(클래스)의 이름을 출력함.
# 감지된 이미지 저장 폴더
save_dir = 'test2_dir'      # 감지된 객체 이미지를 저장할 폴더 이름을 'test2_dir'로 지정함.
os.makedirs(save_dir, exist_ok=True)     # 지정된 폴더가 없으면 새로 생성함. 이미 존재하면 오류를 발생시키지 않음.

cap = cv2.VideoCapture(0)    # 기본 웹캠(0번)을 엶.
if not cap.isOpened():      # 웹캠이 제대로 열리지 않았는지 확인하는 조건문.
    print("웹캠사용불가")
    exit()
else:
    print("웹캠사용가능")
cv2.namedWindow('YOLO 실시간 객체 감지', cv2.WINDOW_NORMAL)  # 'YOLO 실시간 객체 감지'라는 이름의 창을 만듦.
cv2.resizeWindow('YOLO 실시간 객체 감지', 800, 600) # 위에서 만든 창의 크기를 800x600 픽셀로 변경함.

# 중복 저장 방지(3초 내에는 같은 객체 저장 x )

last_saved_time = {}     # 각 객체 라벨별로 마지막으로 저장된 시간을 저장할 딕셔너리임.

while True:         # 무한 루프를 시작함. 웹캠 영상 스트리밍을 계속 처리하기 위함.
    ret, frame = cap.read()          # 웹캠에서 프레임(이미지)을 읽어옴. ret은 성공 여부(True/False), frame은 읽어온 이미지 데이터임.
    if not ret: # 프레임을 읽어오는 데 실패했는지 확인하는 조건문.
        print('프레임을 읽을 수 없어요')
        break   

    results = model(frame, verbose=False)    # 현재 프레임에 대해 YOLO 모델을 실행하여 객체를 감지함. verbose=False는 감지 과정을 콘솔에 자세히 출력하지 않게 함.

    # 특정 객체만 감지에 참여
    allowed_labels =[
        'person','laptop','mouse','keyboard','cell phone','book','clock'
    ]   # 감지할 객체 라벨들을 리스트로 지정함.

    for result in results:  # 감지 결과(results)를 반복함.
        for box in result.boxes:    # 각 결과에 포함된 바운딩 박스 정보를 반복함.
            # 특정 객체만 감지
            label = result.names[int(box.cls[0])]   # 감지된 객체의 이름을 얻음.
            # if label != 'person': continue 사람만 허용
            if label not in allowed_labels: continue    # 만약 감지된 객체 라벨이 `allowed_labels` 리스트에 없으면, 다음 객체로 넘어감.


            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스의 좌표를 정수형으로 추출함.
            label = result.names[int(box.cls[0])]   # 다시 한번 라벨을 얻음.
            confidence = box.conf[0].item()         # 감지된 객체의 신뢰도(확률)를 얻음.

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)    # 프레임 위에 초록색 바운딩 박스를 그림.
            cv2.putText(frame,f'{label} {confidence:.2f}', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)  # 바운딩 박스 위에 라벨과 신뢰도를 텍스트로 표시함.

            # 2초 간격으로 중복 방지 저장
            now = time.time()   # 현재 시간을 초 단위로 얻음.
            last_time = last_saved_time.get(label, 0) # 어떤 객체가 처음으로 감지되면 0을 반환 (해당 라벨의 마지막 저장 시간을 딕셔너리에서 가져옴. 없으면 0을 가져옴.)
            if now - last_time >= 3:         # 현재 시간과 마지막 저장 시간의 차이가 3초 이상인지 확인하는 조건문임.
                filename = f'{label}_{int(now)}.jpg' # 저장할 파일 이름을 '라벨_현재시간.jpg' 형식으로 만듦.
                filepath = os.path.join(save_dir, filename) # 저장할 파일의 전체 경로를 만듦.
                cv2.imwrite(filepath, frame) # 현재 프레임을 지정된 경로에 이미지 파일로 저장함.
                print(f'저장 성공 :{filepath}') # 저장 성공 메시지를 출력함.
                last_saved_time[label] = now # 마지막 저장 시간을 현재 시간으로 업데이트함.
    # 감지된 프레임 화면에 출력
    cv2.imshow('YOLO 실시간 객체 감지', frame)  # 바운딩 박스와 텍스트가 추가된 프레임을 화면에 보여줌.
    key = cv2.waitKey(1) # 1ms 동안 입력 대기. 아무키도 안 누르면 -1 반환
    if key != -1:
        print('눌린 키 :', key, chr(key)) # 눌린 키의 아스키코드 값과 문자를 출력함.
    print('눌린 키 :', key) # 눌린 키의 아스키코드 값을 출력함.
    if key &0xFF == ord('q'):       # ord('a') -> 97 반환 눌린 키가 'q'인지 확인하는 조건문임
        break

# 자원 정리
cap.release()   # 사용중인 카메라 장치(점유) 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫음.

"""
지금 코드는 YOLO와 OpenCV를 이용한 실시간 객체 감지 및 이미지 저장의 기본을 담고 있음 추가 알면 좋은 내용

1. 객체 감지 모델의 원리 심화
YOLO 모델 구조: YOLOv8n.pt는 YOLOv8 모델의 가장 작은 버전. YOLO가 어떻게 입력 이미지를 그리드(grid)로 나누고, 각 셀에서 바운딩 박스와 신뢰도를 예측하는지 그 원리를 이해. 

Backbone, Neck, Head는 딥러닝 모델, 특히 객체 감지 모델의 구조를 이루는 세 가지 주요 구성 요소. YOLOv8 같은 모델이 어떻게 작동하는지 이해하는 데 핵심적인 개념.

Backbone (백본)
백본은 이미지의 특징을 추출하는 역할. 마치 사람의 시각 시스템에서 사물을 인식하기 위해 눈이 정보를 받아들이는 것처럼, 백본은 입력된 이미지에서 중요한 시각적 특징들을 찾아냄. 
예를 들어, 엣지(가장자리), 코너, 색상, 질감 등과 같은 저수준 특징부터, 점점 더 복잡한 형태, 패턴 같은 고수준 특징까지 계층적으로 추출. 
YOLOv8 모델에서는 CSPDarknet53과 같은 구조가 백본으로 사용되어 이미지의 정보를 압축하고 정제하는 역할.

Neck (넥)
넥은 백본이 추출한 다양한 특징들을 하나로 통합하고 융합하는 역할. 
백본은 이미지의 여러 부분에서 다양한 깊이(레벨)의 특징 맵을 생성하는데, 넥은 이 특징 맵들을 서로 연결하여 정보의 흐름을 최적화. 
이는 서로 다른 크기의 객체를 더 잘 감지할 수 있도록 도움. 
예를 들어, 작은 물체(마우스)를 감지하는 데 필요한 세밀한 특징과 큰 물체(사람)를 감지하는 데 필요한 넓은 영역의 특징을 효과적으로 결합. YOLOv8에서는 PANet 같은 구조가 이 넥 역할을 수행하여 정보 손실을 최소화.

Head (헤드)
헤드는 실제로 객체를 예측하는 최종 단계. 넥을 통해 정제된 특징들을 바탕으로, 각 객체의 바운딩 박스 좌표, 클래스(라벨), 신뢰도를 예측. 헤드는 보통 여러 개의 출력 레이어로 구성되며, 각 레이어가 다른 스케일(크기)의 객체를 감지하도록 설계. 
이 덕분에 모델은 이미지에 있는 모든 크기의 객체를 효과적으로 찾아낼 수 있음.


비-최대 억제(Non-Maximum Suppression, NMS): 현재 코드는 여러 개의 바운딩 박스가 한 객체를 감지하는 경우를 모두 처리.
NMS는 여러 박스 중 가장 높은 신뢰도를 가진 박스 하나만 남기고 나머지를 제거하는 기술. 이를 적용하면 중복 감지를 줄이고 더 깔끔한 결과를 얻을 수 있음.

2. OpenCV와 Python 기능 확장
NumPy 배열 조작: OpenCV는 이미지를 NumPy 배열로 다룹니다. 배열 슬라이싱(image[y1:y2, x1:x2])을 능숙하게 사용하면 객체 영역만 잘라내거나, 특정 픽셀을 수정하는 등 다양한 이미지 처리가 가능.

영상 스트림 최적화: 현재 코드는 waitKey(1)로 1ms마다 프레임을 처리합니다. 더 부드러운 영상 처리를 위해 멀티스레딩(multithreading)을 사용해 비디오 프레임 읽기 부분을 별도의 스레드에서 실행.

Matplotlib와 OpenCV의 연동: cv2.cvtColor()를 사용하여 OpenCV(BGR)와 Matplotlib(RGB) 간의 색상 채널 순서 변환을 이해하고, 이를 통해 이미지 시각화가 어떻게 이루어지는지 확인!

3. 프로젝트 활용 및 확장
객체 추적(Object Tracking): 단순히 객체를 감지하는 것을 넘어, 영상에서 특정 객체가 움직이는 경로를 추적하는 기술. 
OpenCV의 내장 트래커(Tracker)나 DeepSORT 같은 라이브러리를 사용해 구현할 수 있음.

데이터베이스 연동: 감지된 객체 정보(라벨, 시간, 좌표 등)를 SQLite나 MongoDB 같은 데이터베이스에 저장하고 관리하면, 나중에 데이터를 분석하거나 활용하기 좋음.

사용자 인터페이스(UI) 개발: 감지된 영상과 객체 정보를 GUI(그래픽 사용자 인터페이스)로 보여주면 더욱 전문적인 프로그램이 됨. PyQt, Tkinter, 또는 Streamlit 라이브러리를 사용해 간단한 UI를 만들어 보기.
"""