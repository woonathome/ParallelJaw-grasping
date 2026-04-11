# Grasping_Mujoco_Sim

`Grasping_Results_SurfaceContact`의 `1-grasping_pairs_feasible_*.html` 결과를 읽어,
MuJoCo 기반 parallel-jaw grasp 동적 테스트를 수행하는 모듈입니다.

## 구현된 시뮬레이션 사양

1. HTML에서 최상위 grasp pair(첫 `pair ...` trace)를 파싱해 object grasp 시뮬레이션 수행
2. grasp 후 jaw 기준 3방향 이동 테스트 수행
   - `depth`: jaw 조임 방향(pad depth 축)
   - `width`: pad width 축
   - `height`: pad height 축
3. 각 방향에서 빠른 이동(속도 기반) 중 object escape 여부 판정
4. 노트북 UI 제공
   - `depth/width/height` 속도
   - 마찰계수 `mu`
   - grasp force(N)
5. MuJoCo native viewer 창 실행 지원
   - 그리퍼 패드 + 물체 상호작용만 렌더링
   - 로봇 암/로봇 베이스 모델은 포함하지 않음

## 파일 구성

- `html_grasp_parser.py`
  - Plotly HTML에서 object mesh + top grasp pair + local grasp 축 추출
- `mujoco_parallel_jaw_sim.py`
  - MuJoCo 모델 생성, grasp/perturb 시뮬레이션, escape 판정, notebook UI
- `__init__.py`
  - 외부 공개 API

## 환경

현재 프로젝트 기준 `sam6d`는 Python `3.9.19`입니다.

```bash
conda run -n sam6d python --version
```

MuJoCo가 없다면 설치:

```bash
conda run -n sam6d pip install mujoco
```

## 빠른 사용

```python
from Grasping_Mujoco_Sim import run_from_html

html_path = "Grasping_Results_SurfaceContact/custom_hh/1-grasping_pairs_feasible_boardmarker_922sol.html"

result = run_from_html(
    html_path=html_path,
    depth_speed=0.8,
    width_speed=0.6,
    height_speed=0.6,
    friction=0.8,
    grasp_force=100.0,
)
print(result["stable_after_grasp"], result["any_escape"])
print(result["perturbation"])
```

뷰어까지 포함해서 실행:

```python
result = run_from_html(
    html_path=html_path,
    friction=0.8,
    grasp_force=100.0,
    depth_speed=0.8,
    width_speed=0.6,
    height_speed=0.6,
    open_viewer=True,       # MuJoCo UI 창 띄움
    realtime_viewer=True,   # 실시간 속도로 재생
)
```

## Notebook UI

```python
from Grasping_Mujoco_Sim import launch_notebook_ui
launch_notebook_ui("Grasping_Results_SurfaceContact/custom_hh/1-grasping_pairs_feasible_boardmarker_922sol.html")
```

## Escape 판정 기준

- 일정 step 이상 finger-object 접촉 상실 (`min_loss_contact_steps`)
- 또는 gripper 기준 object 상대 변위가 임계치 초과 (`escape_distance`)

두 조건 중 하나라도 만족하면 해당 방향 escape로 판정합니다.
