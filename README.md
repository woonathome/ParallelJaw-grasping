# Parallel Gripper Grasping (Planar Patches)

2-finger parallel gripper용 **그리핑 후보 패치 쌍**을 CAD mesh에서 자동으로 찾아내는 알고리즘  
임의의 CAD 모델에서 평면 패치를 추출하고, 법선을 정렬한 뒤, **서로 마주보는 패치 쌍**을 점수화하여 파지 가능성을 평가합니다.

---

## Features

- **Planar patch extraction**: 인접 face를 각도/공면성 기준으로 병합 → 면적 작은 패치 필터링
- **Normal orientation**: mesh 외곽 실루엣 기준으로 법선을 일관되게 외부(또는 내부) 방향으로 정렬
- **Patch pair scoring**:
  - 평행 점수 (정반대 노멀일수록 높음)
  - **Projection overlap ratio** (한 패치 샘플을 다른 패치 평면으로 -n 방향 투영해 내부에 포함되는 비율)
  - 최종 점수 = 병합 가중합
- **Opening width 계산**: 두 패치 중심 좌표 기반 (추후 최소 거리 기반으로 개선 예정)
- **Feasibility check** (pad collision): 폭/높이/깊이 지정된 pad OBB를 생성해 mesh와 충돌 여부 확인
- **Interactive visualization**: Plotly 기반으로 패치, 페어, pad 가시화 및 HTML 저장

---

## Install

```anaconda prompt
pip install -U numpy trimesh open3d plotly
# optional for fast ray queries and image export
pip install pyembree kaleido
