# 인스턴스 세그멘테이션: 욜로가 직접 내 주는 결과. 객체마다 마스크가 따로 존재
# 의미론적 세그멘테이션: 이미지 내의 픽셀단위로 '이 픽셀은 어느 클래스에 속한다' 만 표현

import os
from tabnanny import verbose
from typing import Annotated
# 1. 초기 설정 및 모델 로딩
# from YOLO_exam.yolotest8seg import IMG_PATH
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2, numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

IMG_PATH = 'animal2.jpg'
OUT_DIR = 'seg_out2'
os.makedirs(OUT_DIR, exist_ok=True)     # 디렉토리가 이미 존재해도 오류를 발생시키지 않음

model = YOLO('yolov8n-seg.pt')
#  2. 객체 탐지 및 마스크 추출
im_bgr = cv2.imread(IMG_PATH)    # 이미지를 BGR(OpenCV 기본)로 읽고, 나중에 imshow()를 위해 RGB로 변환
im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
H, W = im_bgr.shape[:2]         # 이미지의 높이와 너비를 가져옴

res = model(im_bgr, verbose=False)[0]   # 모델을 실행하여 결과. `verbose=False`는 자세한 출력 없이 결과를 반환
annotated = res.plot()      #YOLO가 감지한 경계 상자와 마스크를 이미지에 시각화하여 반환
cv2.imwrite(os.path.join(OUT_DIR, 'seg_result.jpg'), annotated)

# pythorch tensor -> numpy 배열로 변환
has_masks = (res.masks is not None)

if has_masks:
    masks_np = res.masks.data.cpu().numpy() #객체별 빅셀 마스크
    boxes_np = res.boxes.xyxy.cpu().numpy() #객체별 경계박스 좌표 바운딩 박스
    confs_np = res.boxes.conf.cpu().numpy() #객체별 신뢰도 점수
    classes_np = res.boxes.cls.cpu().numpy().astype(int) #객체별 클래스 번호

else:
    masks_np = boxes_np = confs_np = classes_np = None


# 3. 마스크 오버레이 및 객체 배경 제거
overlay = im_bgr.copy()
if has_masks:
    for m in masks_np:
        m_bin = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
        color_mask = np.zeros_like(overlay)
        color_mask[m_bin] = (0, 255, 0) # 객체 마스크 픽셀만 초록색으로 채움
        overlay = cv2.addWeighted(overlay, 0.5, color_mask, 0.4, 0.0)  # 원본 + 칼라마스크 

cv2.imwrite(os.path.join(OUT_DIR, 'seg_overlay.jpg'), annotated)

# 객체별 배경 제거 
crops_dir = os.path.join(OUT_DIR, 'seg_drops')
os.makedirs(crops_dir, exist_ok=True)
# `masks_full`은 모든 객체의 마스크를 원본 크기로 확대하여 하나의 배열로 만든 것
if has_masks and len(masks_np) > 0:
    masks_full = np.stack([
        cv2.resize(m, (W, H), cv2.INTER_NEAREST) > 0.5 for m in masks_np
    ], axis = 0)   # 각 객체별 (H, W) 마스크를 (N, H, W)로 변환

    # 탐지된 객체의 배경을 제거해 png 파일로 잘라내기
    for i,(m_full, box, cls_id, conf) in enumerate(zip(masks_full, boxes_np, classes_np, confs_np)):    # m_full은 불리언 마스크, box는 경계 상자, cls_id는 클래스 번호, conf는 신뢰도
        x1,y1,x2,y2 = map(int, box)
        x1,y1 = max(0, x1), max(0, y1)  #좌상단 좌표가 이미지 밖으로 나가면 0으로 보정
        x2,y2 = min(W, x2), min(H, y2)  #우하단 좌표가 이미지 밖으로 나가면 0으로 보정
        if x2 <= x1 or y2 <= y1:
            continue

        # 원본 이미지에서 경계 상자 영역을 잘라
        crop_bgr = im_bgr[y1:y2, x1:x2]
        crop_mask = (m_full[y1:y2, x1:x2]*255).astype(np.uint8)
        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)  #BGR -> BGRA
        crop_bgra[..., :, 3] = crop_mask # 알파채널에 마스크 적용 -> 배경 투명, 객체 불투명
        # 클래스 이름 또는 id 얻기 
        name =  model.names[int(cls_id)] if hasattr(model, 'names') else str(cls_id)
        cv2.imwrite(os.path.join(crops_dir, f'crop_{i}_{name}_{conf:.2f}.png'), crop_bgra)
        
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# 의미론적 분할(semantic segmentation)
sem_canvas = np.zeros((H,W,3), dtype=np.uint8)   # 최종 색상 이미지
conf_map = np.zeros((H,W), dtype=float)   # 선택된 인스턴스의 신뢰도를 기록할 맵
# 클래스 ID에 따라 고유한 색상을 반환하는 함수
def class_color(c:int):
    return ((23 * c) % 256, (19 * c) % 256, (77 * c) % 256)

#  모든 마스크를 순회하며 최종 의미론적 분할 이미지 만듦
if has_masks and len(masks_np) > 0:
    # `update`는 현재 마스크가 존재하고, 해당 픽셀의 신뢰도가 `conf_map`에 기록된 신뢰도보다 높을 때 True.
    # 이를 통해 겹치는 객체들 중 가장 신뢰도가 높은 마스크만 남김.
    # m_full은 이미 masks_full에서 불리언 배열로 변환되었으므로, 바로 & 연산이 가능.
    for m_full, box, cls_id, conf in zip(masks_full, boxes_np, classes_np, confs_np):
        color = class_color(int(cls_id))
        update = m_full & (conf > conf_map)
        # update가 True인 픽셀에만 해당 객체의 색상과 신뢰도를 기록
        sem_canvas[update] = color
        conf_map[update] = conf

cv2.imwrite(os.path.join(OUT_DIR, 'seg_semantic.png'), sem_canvas)

"""
핵심: 인스턴스 세그멘테이션(객체마다 마스크) 결과를 의미론적 세그멘테이션(픽셀마다 클래스)으로 변환하는 과정. 
conf_map을 이용해 겹치는 영역에서 가장 신뢰도가 높은 객체의 마스크만 선택하는 로직
"""