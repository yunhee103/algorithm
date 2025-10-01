import os, cv2, numpy as np
from ultralytics import YOLO


IMG_PATH = 'image1.jpg'
OUT_DIR = 'seg_out'
os.makedirs(OUT_DIR, exist_ok=True)

# 이미지 읽기
im = cv2.imread(IMG_PATH)
# assert im is not None, f'이미지 읽기 실패 : {IMG_PATH}'

H, W = im.shape[:2]
print(f"입력 이미지 크기: {H} x {W}")

# model
model = YOLO('yolov8n-seg.pt')
res = model(im)[0]
# print(res)    # boxes, masks, names, array ...

cv2.imwrite(os.path.join(OUT_DIR, 'anno1.jpg'), res.plot())
# res.plot() : 원본 이미지에 바운딩박스, 레이블, 신뢰도, 세그먼테이션 마스크를 표시하여 BGR 이미지로 반환
# C:\work\pysou\seg_out\anno1.jpg

# 마스크가 없으면 작업 x
if res.masks is None or len(res.masks.data) == 0:
    print('마스크 없음')
    raise SystemExit

m_small = res.masks.data.cpu().numpy()
# print(m_small)

masks = np.stack([
        cv2.resize(m, (W, H), cv2.INTER_NEAREST) > 0.5 for m in m_small
], axis = 0)   # 각 객체별 (H, W) 마스크를 (N, H, W)로 변환
# print(masks)

# 세그 전 단계 : 마스크 프리뷰
# 마스크가 같은 위치 픽셀에 대해 객체중 하나라도 1(True)이면 n개 마스크를 or 연산으로 합침
mask_union = (masks.any(axis = 0).astype(np.uint) * 255)
cv2.imwrite(os.path.join(OUT_DIR, 'mask_preview.jpg'), mask_union)

# 최종 세그멘테이션 : 컬러 오버레이 + 외곽선
def color(i):
    return ((37 * i) % 256, (17 * i) % 256, (91 * i) % 256)

final = im.copy()   # 직접 원본에 그리기 x, 복사본에서 작업
blend = np.zeros_like(im)   # 오버레이 색 채우기 캔버스

# 컬러 오버레이(blend)는 객체 내부를 색칠하고 경계선 그리기
for i, m in enumerate(masks):
    blend[m] = color(i)    # 마스크 영역의 칠해질 색 
    cnts, _ = cv2.findContours(   # 마스크 외곽선 
        (m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # 0 ~ 1  →  0 ~ 255로 이진화   가장 바깥쪽 외곽선        꼭지점 단순화  
    )
    cv2.drawContours(final, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)

# 반투명 합성
final = cv2.addWeighted(final, 1.0, blend, 0.45, 0.0)
cv2.imwrite(os.path.join(OUT_DIR, 'final_preview.jpg'), final)
cv2.imshow('final segmentation', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
