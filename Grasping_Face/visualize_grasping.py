import numpy as np
import plotly.graph_objects as go
import trimesh
import colorsys
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import asdict
from scipy.spatial import ConvexHull
from plotly.subplots import make_subplots

# grasping module
from . import grasping as gf

# -------------------------------
# Helpers (Visualization Specific)
# -------------------------------
def _get_feasible_reports(reports: List[Dict[str, Any]], max_show: Optional[int]) -> List[Dict[str, Any]]:
    feasible_reports = [r for r in reports if r.get("feasible")]
    if max_show is not None:
        feasible_reports = feasible_reports[:max_show]
    return feasible_reports


def mesh3d_from_trimesh(
    m: trimesh.Trimesh, 
    face_indices: Optional[Union[List[int], np.ndarray, set]] = None, 
    color: str = "lightgray", 
    opacity: float = 0.3, 
    name: str = "mesh"
) -> go.Mesh3d:
    """Trimesh 객체를 Plotly Mesh3d 객체로 변환"""
    if face_indices is None:
        faces = m.faces
    else:
        faces = m.faces[np.asarray(list(face_indices), dtype=int)]
        
    x, y, z = m.vertices.T
    i, j, k = faces.T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.2, roughness=0.9),
        lightposition=dict(x=0, y=0, z=1),
        name=name, showlegend=True
    )

def boundary_edges_trace(
    mesh: trimesh.Trimesh, 
    face_idx: Union[List[int], np.ndarray, set], 
    color: str = "#166534", 
    width: int = 1, 
    name: str = "boundary"
) -> Optional[go.Scatter3d]:
    """Patch 경계선(외곽 edge만)을 Scatter3d로 표현"""
    F = mesh.faces[np.asarray(list(face_idx), dtype=int)]
    # edge -> count
    edges: Dict[Tuple[int, int], int] = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edges[e] = edges.get(e, 0) + 1
            
    # 경계(edge count==1)만 라인화
    lines = []
    for (u, v), c in edges.items():
        if c == 1:
            p1, p2 = mesh.vertices[u], mesh.vertices[v]
            lines.extend([p1, p2, [None, None, None]])  # 분리용 None
            
    if not lines:
        return None
        
    L = np.array(lines, dtype=object) # float 대신 object (None 포함)
    # None 필터링 주의: Plotly는 None으로 선을 끊음. 좌표값만 추출시에는 주의 필요하나 여기선 바로 전달
    
    # 좌표 분리 (None 처리를 위해 리스트 컴프리헨션 사용 권장되나, numpy object array도 plotly가 처리함)
    # 안전하게 변환:
    x_coords = [p[0] if p is not None else None for p in lines]
    y_coords = [p[1] if p is not None else None for p in lines]
    z_coords = [p[2] if p is not None else None for p in lines]

    return go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode="lines", line=dict(color=color, width=width),
        name=name, showlegend=False
    )

def hsv(i: int, n: int, s: float = 0.6, v: float = 0.95) -> str:
    """인덱스 기반 HSV 색상 생성"""
    h = (i % n) / max(1, n)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def hsv_from_value(
    value: float, 
    min_val: float, 
    max_val: float, 
    s: float = 0.9, 
    v: float = 0.95
) -> str:
    """
    값의 크기에 따른 색상 매핑
    - min_val -> 파란색 (Hue=0.7)
    - max_val -> 빨간색 (Hue=0.0)
    """
    val_range = max_val - min_val
    if val_range < 1e-6: # 모든 값이 같으면 중간색(마젠타)
        norm_val = 0.5
    else: # 값을 0.0 ~ 1.0 사이로 정규화
        norm_val = (value - min_val) / val_range
    # Hue 스케일: 0.0(정규화값) -> 0.7(파랑), 1.0(정규화값) -> 0.0(빨강)
    hue = 0.7 - (norm_val * 0.7)
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def cones_for_normals(
    patches: List[Dict[str, Any]],
    mesh_bounds: np.ndarray,
    normal_scale: float = 20.0,
    min_len_ratio: float = 0.1,
    color: str = "#000000",
    name: str = "normals",
) -> go.Cone:
    """Normal 벡터를 Cone 형태로 시각화"""
    (bmin, bmax) = mesh_bounds
    extent = float(np.linalg.norm(np.asarray(bmax) - np.asarray(bmin))) or 1.0
    bbox_min_len = max(min_len_ratio * extent, 1e-6)

    cx, cy, cz, ux, uy, uz, lens = [], [], [], [], [], [], []
    for p in patches:
        c = np.asarray(p["centroid"], float)
        n = np.asarray(p["normal"], float)
        n = n / (np.linalg.norm(n) + 1e-12)
        a = float(p.get("area", 0.0))
        nlen = max(bbox_min_len, 0.5 * np.sqrt(max(a, 0.0)))  # 글로벌+로컬
        lens.append(nlen)

        cx.append(c[0]); cy.append(c[1]); cz.append(c[2])
        ux.append(n[0]*nlen); uy.append(n[1]*nlen); uz.append(n[2]*nlen)

    med = float(np.median(lens)) if lens else bbox_min_len
    sizeref = max(med / max(normal_scale, 1e-6), 1e-9)

    return go.Cone(
        x=cx, y=cy, z=cz, u=ux, v=uy, w=uz,
        sizemode="raw", sizeref=sizeref,
        anchor="tail", colorscale=[[0, color], [1, color]],
        showscale=False, name=name, opacity=1.0,
    )

def frame_traces(
    H: np.ndarray, 
    name: str, 
    scale: float = 20.0, 
    legendgroup: Optional[str] = None, 
    showlegend: bool = False
) -> List[go.Scatter3d]:
    """4x4 pose 행렬 H 기준 좌표축(x=red, y=green, z=blue) 시각화"""
    o = H[:3, 3]
    x_axis = o + H[:3, 0] * scale
    y_axis = o + H[:3, 1] * scale
    z_axis = o + H[:3, 2] * scale
    traces = []
    traces.append(go.Scatter3d(x=[o[0],x_axis[0]], y=[o[1],x_axis[1]], z=[o[2],x_axis[2]],
                               mode="lines", line=dict(color="red", width=6),
                               name=f"{name}-x", legendgroup=legendgroup, showlegend=showlegend))
    traces.append(go.Scatter3d(x=[o[0],y_axis[0]], y=[o[1],y_axis[1]], z=[o[2],y_axis[2]],
                               mode="lines", line=dict(color="green", width=6),
                               name=f"{name}-y", legendgroup=legendgroup, showlegend=showlegend))
    traces.append(go.Scatter3d(x=[o[0],z_axis[0]], y=[o[1],z_axis[1]], z=[o[2],z_axis[2]],
                               mode="lines", line=dict(color="blue", width=6),
                               name=f"{name}-z", legendgroup=legendgroup, showlegend=showlegend))
    return traces

# -------------------------------
# 1) 추출된 패치 시각화
# -------------------------------
def visualize_merged_patches_plotly(
    result: Dict[str, Any],
    show_silhouette: bool = True,
    show: bool = False
) -> go.Figure:
    """
    result: grasping.compute_best_patch_pairs(...) 출력 dict
    """
    mesh_sil: trimesh.Trimesh = result["mesh_quad"]
    mesh_p: trimesh.Trimesh = result["mesh_patches"]
    patches: List[Dict[str, Any]] = result["patches"]

    # 범위/스케일
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_sil.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin, ymax-ymin, zmax-zmin)

    # 색/스타일
    patch_fill_color  = "#a7f3d0"  # 옅은 초록
    edge_color        = "#166534"  # 진한 초록(테두리)
    normal_color      = "#666666"  # 검정

    traces = []
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh_sil, None, color="#cfcfcf", opacity=0.12, name="silhouette"))

    # 모든 패치: 동일 색 + 경계선 + 검정 normal
    for p in patches:
        fidx = set(p["face_indices"])
        traces.append(mesh3d_from_trimesh(mesh_p, fidx, color=patch_fill_color, opacity=0.6, name=f"patch {p['id']}"))
        edge = boundary_edges_trace(mesh_p, fidx, color=edge_color, width=2, name=f"edge {p['id']}")
        if edge is not None:
            traces.append(edge)

    # normals(검정)
    traces.append(cones_for_normals(patches, mesh_bounds=mesh_sil.bounds, normal_scale=50.0,
                                     min_len_ratio=0.1, color=normal_color, name="normals"))
    
    # COM 넣기
    com = mesh_sil.center_mass
    traces.append(go.Scatter3d(x=[com[0]],y=[com[1]],z=[com[2]],
                               mode='markers', marker=dict(size=5,color='black',symbol='diamond',opacity=0.5),
                               name='Center of Mass'))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Patches (count={len(patches)})",
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# -------------------------------
# 2) patch pair(top_k) 시각화
# -------------------------------
def visualize_pairs_centroid_lines(
    result: Dict[str, Any],
    show_silhouette: bool = True,
    patch_opacity: float = 0.3,
    max_pairs_show: Optional[int] = 200,
    show: bool = False
) -> go.Figure:
    """
    result['top_k']의 모든 pair 표시
    """
    mesh_sil: trimesh.Trimesh = result["mesh_quad"]
    mesh_p: trimesh.Trimesh = result["mesh_patches"]
    patches = result["patches"]
    cands = result.get("top_k", [])
    if max_pairs_show is not None:
        cands = cands[:max_pairs_show]
    if not cands:
        raise ValueError("result['top_k'] is empty.")

    # id -> patch
    pmap = {p["id"]: p for p in patches}

    # 범위
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_p.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces = []
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh_sil, None, color="#cfcfcf", opacity=0.12, name="silhouette"))

    for k, cand in enumerate(cands):
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in pmap or pid_j not in pmap:
            continue
        pi, pj = pmap[pid_i], pmap[pid_j]
        fi, fj = set(pi["face_indices"]), set(pj["face_indices"])
        col = hsv(k, len(cands), s=0.55, v=0.95)          # pair 공통색
        col_edge = hsv(k, len(cands), s=0.85, v=0.7)      # 경계 진한색

        # patches (같은 색, 옅게)
        traces.append(mesh3d_from_trimesh(mesh_p, fi, color=col, opacity=patch_opacity, name=f"pair{k}: patch {pid_i}"))
        traces.append(mesh3d_from_trimesh(mesh_p, fj, color=col, opacity=patch_opacity, name=f"pair{k}: patch {pid_j}"))

        # 경계선
        e1 = boundary_edges_trace(mesh_p, fi, color=col_edge, width=3, name=f"pair{k}: edge i")
        e2 = boundary_edges_trace(mesh_p, fj, color=col_edge, width=3, name=f"pair{k}: edge j")
        if e1 is not None: traces.append(e1)
        if e2 is not None: traces.append(e2)

        # centroid line (COM of patch surfaces ≈ face centroid 평균 사용)
        ci = np.asarray(pi["centroid"], float)
        cj = np.asarray(pj["centroid"], float)
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines+markers",
            line=dict(width=6, color=col_edge),
            marker=dict(size=3, color=col_edge),
            name=f"pair{k}: centroid line"
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Top-k Pairs (k={len(cands)}) — patches (light) + centroid lines",
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# -------------------------------
# 3) feasible patch pair 시각화
# -------------------------------
# pad mesh 있는 버전
def visualize_feasible_pairs_pads(
    result: Dict[str, Any], 
    reports: List[Dict[str, Any]],
    show_silhouette: bool = True,
    pad_w: float = gf.PadParams.pad_w,
    pad_h: float = gf.PadParams.pad_h,
    pad_d: float = gf.PadParams.pad_d,
    max_reports_show: Optional[int] = 300,
    show: bool = False
) -> go.Figure:
    
    mesh: trimesh.Trimesh = result.get("mesh_quad")
    mesh_patches: trimesh.Trimesh = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    dists = [r["dist"] for r in reports if r.get("feasible")] # 무게중심 - pair 거리로 색상 코딩
    
    feasible_reports = _get_feasible_reports(reports, max_reports_show)
    ok_idx = [r["pair_index"] for r in feasible_reports]
    dists = [r["dist"] for r in feasible_reports]

    if not dists:
        colval_min, colval_max = 0, 1
    else:
        colval_min, colval_max = min(dists), max(dists) # 무게중심 - pair 거리로 색상 코딩

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.6, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        report = feasible_reports[k]
        yaw_deg = reports[k]['feasible_yaw']
        yaw_deg = report.get("feasible_yaw", yaw_deg)
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        fi, fj = report["face_i"], report["face_j"]
        ni = gf.unit(np.asarray(pi["normal"], float))
        nj = gf.unit(np.asarray(pj["normal"], float))
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)
        
        # gf.make_rot_pad_box_at_patch 사용 (clearance_out=0, lift_mm=0으로 설정하여 단순 시각화)
        box_i = gf.make_rot_pad_box_at_patch(
            {"centroid": ci, "normal": ni}, pad_w, pad_h, pad_d, 
            clearance_out=0.0, lift_mm=0.0, yaw_deg=yaw_deg
        )
        box_j = gf.make_rot_pad_box_at_patch(
            {"centroid": cj, "normal": nj}, pad_w, pad_h, pad_d, 
            clearance_out=0.0, lift_mm=0.0, yaw_deg=-yaw_deg
        )

        # 색상 설정
        col = hsv_from_value(dists[k], min_val=colval_min, max_val=colval_max)

        # legendgroup으로 두 pad를 하나의 토글 그룹에 묶기
        lg = f"pair {idx} - faces {fi, fj} - yaw {yaw_deg}"
        # proxy(범례 핸들) – 클릭 시 그룹 전체 토글
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True))
        # 그리퍼 패드 시각화
        # pad_i
        x,y,z = box_i.vertices.T; I,J,K = box_i.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_i", legendgroup=lg, showlegend=False))
        # pad_j
        x,y,z = box_j.vertices.T; I,J,K = box_j.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_j", legendgroup=lg, showlegend=False))
        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5),
            name=f"{lg} - centerline", legendgroup=lg, showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads only (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # ← 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# pad mesh 없는 버전 (Yaw 정보만 표시)
def visualize_feasible_pairs_with_yaw(
    result: Dict[str, Any], 
    reports: List[Dict[str, Any]],
    show_silhouette: bool = True,
    pad_w: float = gf.PadParams.pad_w,
    pad_h: float = gf.PadParams.pad_h,
    pad_d: float = gf.PadParams.pad_d,
    max_reports_show: Optional[int] = 300,
    show: bool = False
) -> go.Figure:
    
    mesh: trimesh.Trimesh = result.get("mesh_quad")
    mesh_patches: trimesh.Trimesh = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    dists = [r["dist"] for r in reports if r.get("feasible")]
    
    feasible_reports = _get_feasible_reports(reports, max_reports_show)
    ok_idx = [r["pair_index"] for r in feasible_reports]
    dists = [r["dist"] for r in feasible_reports]

    if not dists:
        colval_min, colval_max = 0, 1
    else:
        colval_min, colval_max = min(dists), max(dists)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.6, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        report = feasible_reports[k]
        yaw_deg = reports[k]['feasible_yaw']
        yaw_deg = report.get("feasible_yaw", yaw_deg)
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        
        # patches 정보는 필요하지만 여기선 좌표 계산에만 쓰임
        # pi = patches[pid_i]; pj = patches[pid_j]

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        fi, fj = report["face_i"], report["face_j"]
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)

        # 색상 설정
        col = hsv_from_value(dists[k], min_val=colval_min, max_val=colval_max)

        # legendgroup으로 두 pad를 하나의 토글 그룹에 묶기
        lg = f"pair {idx} - faces {fi, fj} - yaw {yaw_deg}"
        # proxy(범례 핸들) – 클릭 시 그룹 전체 토글
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True))

        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5),
            name=f"{lg} - centerline", legendgroup=lg, showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads only (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # ← 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

def visualize_feasible_pairs_with_cylinder(
    result: Dict[str, Any], 
    reports: List[Dict[str, Any]],
    show_silhouette: bool = True,
    pad_w: float = gf.PadParams.pad_w,
    pad_h: float = gf.PadParams.pad_h,
    pad_d: float = gf.PadParams.pad_d,
    lift_mm: float = 2.0,
    max_reports_show: Optional[int] = 300,
    show: bool = False,
    vis_cyl: bool = True
) -> go.Figure:

    mesh: trimesh.Trimesh = result.get("mesh_quad")
    mesh_patches: trimesh.Trimesh = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    dists = [r["dist"] for r in reports if r.get("feasible")]
    
    feasible_reports = _get_feasible_reports(reports, max_reports_show)
    ok_idx = [r["pair_index"] for r in feasible_reports]
    dists = [r["dist"] for r in feasible_reports]

    if not dists:
        colval_min, colval_max = 0, 1
    else:
        colval_min, colval_max = min(dists), max(dists)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.6, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        report = feasible_reports[k]
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        fi, fj = report["face_i"], report["face_j"]
        ni = gf.unit(np.asarray(pi["normal"], float))
        nj = gf.unit(np.asarray(pj["normal"], float))
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)

        # 색상 설정
        col = hsv_from_value(dists[k], min_val=colval_min, max_val=colval_max)

        # legendgroup 생성
        lg = f"Pair {idx} | Faces ({fi}, {fj})"

        # --- 원기둥 생성 및 추가 ---
        if vis_cyl:
            # gf.make_pad_cylinder_at_patch 사용 (clearance_out=0.0 설정)
            cyl_i = gf.make_pad_cylinder_at_patch(
                {"centroid": ci, "normal": ni}, pad_w / 2, pad_h / 2, pad_d, 
                clearance_out=0.0, lift_mm=lift_mm
            )
            cyl_j = gf.make_pad_cylinder_at_patch(
                {"centroid": cj, "normal": nj}, pad_w / 2, pad_h / 2, pad_d, 
                clearance_out=0.0, lift_mm=lift_mm
            )

            # 원기둥 i trace 추가
            x,y,z = cyl_i.vertices.T; I,J,K = cyl_i.faces.T
            traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                    color=col, opacity=0.2,
                                    name=f"{lg} - cyl_i", legendgroup=lg, showlegend=False))
            # 원기둥 j trace 추가
            x,y,z = cyl_j.vertices.T; I,J,K = cyl_j.faces.T
            traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                    color=col, opacity=0.2,
                                    name=f"{lg} - cyl_j", legendgroup=lg, showlegend=False))

        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5), # 중심선은 원래 색상
            name=f"{lg}", legendgroup=lg, showlegend=True
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Swept Volumes (Cylinders) (count={len(ok_idx)})", # 제목 변경
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),
        scene=dict(
            xaxis=dict(range=[cx - max_range, cx + max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy - max_range, cy + max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz - max_range, cz + max_range], showgrid=False, zeroline=False),
            aspectmode="cube" # aspectmode 변경 (데이터 비율에 맞게)
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# pad mesh 없는 버전
def visualize_feasible_pairs(
    result: Dict[str, Any], 
    reports: List[Dict[str, Any]],
    show_silhouette: bool = True,
    max_reports_show: Optional[int] = 300,
    show: bool = False
) -> go.Figure:
    
    mesh: trimesh.Trimesh = result.get("mesh_quad")
    mesh_patches: trimesh.Trimesh = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    dists = [r["dist"] for r in reports if r.get("feasible")]
    
    feasible_reports = _get_feasible_reports(reports, max_reports_show)
    ok_idx = [r["pair_index"] for r in feasible_reports]
    dists = [r["dist"] for r in feasible_reports]

    if not dists:
        colval_min, colval_max = 0, 1
    else:
        colval_min, colval_max = min(dists), max(dists)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.6, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        report = feasible_reports[k]
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        # pi, pj = patches[pid_i], patches[pid_j]

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        fi, fj = report["face_i"], report["face_j"]
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)

        # 색상 설정
        col = hsv_from_value(dists[k], min_val=colval_min, max_val=colval_max)

        # legendgroup으로 두 pad를 하나의 토글 그룹에 묶기
        lg = f"pair {idx} - faces {fi, fj}"
        # proxy(범례 핸들) – 클릭 시 그룹 전체 토글
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True))

        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5),
            name=f"{lg} - centerline", legendgroup=lg, showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads only (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # ← 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


# -------------------------------
# 4) feasible patch pair + 그리퍼 좌표계
# -------------------------------
def visualize_feasible_pairs_pads_gripper(
    result: Dict[str, Any], 
    reports: List[Dict[str, Any]],
    show_silhouette: bool = True,
    pad_w: float = gf.PadParams.pad_w,
    pad_h: float = gf.PadParams.pad_h,
    pad_d: float = gf.PadParams.pad_d,
    max_reports_show: Optional[int] = 300,
    show: bool = False
) -> go.Figure:
    
    mesh: trimesh.Trimesh = result.get("mesh_quad")
    mesh_patches: trimesh.Trimesh = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    dists = [r["dist"] for r in reports if r.get("feasible")] # 무게중심 - pair 거리로 색상 코딩
    
    feasible_reports = _get_feasible_reports(reports, max_reports_show)
    ok_idx = [r["pair_index"] for r in feasible_reports]
    dists = [r["dist"] for r in feasible_reports]

    if not dists:
        colval_min, colval_max = 0, 1
    else:
        colval_min, colval_max = min(dists), max(dists) # 무게중심 - pair 거리로 색상 코딩

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.6, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        report = feasible_reports[k]
        yaw_deg = reports[k].get('feasible_yaw', 0.0)
        yaw_deg = report.get("feasible_yaw", yaw_deg)
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        fi, fj = report["face_i"], report["face_j"]
        ni = gf.unit(np.asarray(pi["normal"], float))
        nj = gf.unit(np.asarray(pj["normal"], float))
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)
        
        # gf.make_rot_pad_box_at_patch 사용
        box_i = gf.make_rot_pad_box_at_patch(
            {"centroid": ci, "normal": ni}, pad_w, pad_h, pad_d, 
            clearance_out=0.0, lift_mm=0.0, yaw_deg=yaw_deg
        )
        box_j = gf.make_rot_pad_box_at_patch(
            {"centroid": cj, "normal": nj}, pad_w, pad_h, pad_d, 
            clearance_out=0.0, lift_mm=0.0, yaw_deg=-yaw_deg
        )

        # 색상 설정
        col = hsv_from_value(dists[k], min_val=colval_min, max_val=colval_max)

        # legendgroup 묶기
        lg = f"pair {idx}"
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True
        ))
        # 그리퍼 패드 시각화
        # pad_i
        x,y,z = box_i.vertices.T; I,J,K = box_i.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_i", legendgroup=lg, showlegend=False))
        # pad_j
        x,y,z = box_j.vertices.T; I,J,K = box_j.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_j", legendgroup=lg, showlegend=False))

        # gripper frame using gf.build_gripper_pose_obj
        H_grip, stroke = gf.build_gripper_pose_obj(
            {"centroid": ci, "normal": ni}, 
            {"centroid": cj, "normal": nj}, 
            yaw_deg
        )
        
        traces += frame_traces(H_grip, lg, scale=15.0, legendgroup=lg, showlegend=False)

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads + Gripper Frames (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


# -------------------------------
# 기타 임시 함수
# -------------------------------
def visualize_mesh_with_edges(
    mesh: trimesh.Trimesh, 
    color: str = "#166534", 
    edge_color: str = "#166534", 
    opacity: float = 0.5,
    edge_width: int = 4, 
    show: bool = True
) -> go.Figure:
    """
    주어진 trimesh.Trimesh를 단순히 시각화.
    """
    # bounding box로 범위 계산
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin, ymax-ymin, zmax-zmin)

    traces = []

    # 표면 (Mesh3d)
    x, y, z = mesh.vertices.T
    I, J, K = mesh.faces.T
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color=color, opacity=opacity,
        name="mesh"
    ))

    # edge (Wireframe)
    edges = mesh.edges_unique
    verts = mesh.vertices
    xe, ye, ze = [], [], []
    for (v0, v1) in edges:
        xe += [verts[v0][0], verts[v1][0], None]
        ye += [verts[v0][1], verts[v1][1], None]
        ze += [verts[v0][2], verts[v1][2], None]
    traces.append(go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode="lines",
        line=dict(color=edge_color, width=edge_width),
        name="edges"
    ))

    # figure
    fig = go.Figure(traces)
    fig.update_layout(
        title="Mesh with Edges",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


def visualize_report_faces(
    result: Dict[str, Any], 
    report: Dict[str, Any], 
    show_silhouette: bool = True, 
    show: bool = True
) -> go.Figure:
    """
    단일 report에 대한 face 시각화
    """
    mesh_sil: trimesh.Trimesh = result["mesh_quad"]
    mesh_p: trimesh.Trimesh = result["mesh_patches"]
    patches  = {p["id"]: p for p in result["patches"]}

    pid_i, pid_j = report["patch_i"], report["patch_j"]
    fi_sel, fj_sel = int(report["face_i"]), int(report["face_j"])

    pi, pj = patches[pid_i], patches[pid_j]

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_sil.bounds
    cx, cy, cz   = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range    = max(xmax-xmin, ymax-ymin, zmax-zmin)

    traces = []
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh_sil, None,
                                          color="#cfcfcf", opacity=0.15,
                                          name="silhouette"))

    # Patch i (연두색)
    fidx_i = set(pi["face_indices"])
    traces.append(mesh3d_from_trimesh(mesh_p, fidx_i, color="#99ff99", opacity=0.5,
                                      name=f"patch {pid_i} faces"))
    edge_i = boundary_edges_trace(mesh_p, fidx_i, color="#008800", width=2,
                                  name=f"patch {pid_i} edges")
    if edge_i is not None:
        traces.append(edge_i)

    # Patch j (하늘색)
    fidx_j = set(pj["face_indices"])
    traces.append(mesh3d_from_trimesh(mesh_p, fidx_j, color="#99ccff", opacity=0.5,
                                      name=f"patch {pid_j} faces"))
    edge_j = boundary_edges_trace(mesh_p, fidx_j, color="#004488", width=2,
                                  name=f"patch {pid_j} edges")
    if edge_j is not None:
        traces.append(edge_j)

    # 강조 face_i (빨강)
    traces.append(mesh3d_from_trimesh(mesh_p, {fi_sel}, color="#ff0000", opacity=1.0,
                                      name=f"face {fi_sel} (patch {pid_i})"))

    # 강조 face_j (파랑)
    traces.append(mesh3d_from_trimesh(mesh_p, {fj_sel}, color="#0000ff", opacity=1.0,
                                      name=f"face {fj_sel} (patch {pid_j})"))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Report pair {report['pair_index']} (patch {pid_i}, {pid_j})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")
    return fig


def visualize_patch_pair_faces(
    result: Dict[str, Any], 
    report: Dict[str, Any], 
    show: bool = True
) -> go.Figure:
    """
    result: grasping.compute_best_patch_pairs(...) 출력 dict
    report: reports[k] (단일 pair)
    """
    mesh_sil: trimesh.Trimesh = result["mesh_quad"]
    mesh_p: trimesh.Trimesh = result["mesh_patches"]
    patches = {p["id"]: p for p in result["patches"]}
    
    pid_i, pid_j = report["patch_i"], report["patch_j"]
    pi, pj = patches[pid_i], patches[pid_j]

    verts = np.asarray(mesh_p.vertices)
    faces = np.asarray(mesh_p.faces)

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_p.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin, ymax-ymin, zmax-zmin)

    traces = []

    traces.append(mesh3d_from_trimesh(mesh_sil, None,
                                        color="#cfcfcf", opacity=0.15,
                                        name="silhouette"))

    # patch i faces (빨강 계열)
    for fidx in pi["face_indices"]:
        tri = faces[fidx]
        x,y,z = verts[tri].T
        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0], j=[1], k=[2],
            color="#000000",
            opacity=0.6,
            name=f"patch {pid_i} face {fidx}",
            legendgroup=f"patch {pid_i} face {fidx}",
            showlegend=True
        ))

    # patch j faces (파랑 계열)
    for fidx in pj["face_indices"]:
        tri = faces[fidx]
        x,y,z = verts[tri].T
        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0], j=[1], k=[2],
            color="#000000",
            opacity=0.6,
            name=f"patch {pid_j} face {fidx}",
            legendgroup=f"patch {pid_j} face {fidx}",
            showlegend=True
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Patch pair {pid_i} & {pid_j} (faces)",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


def visualize_frames(
    H_dict: Dict[str, np.ndarray], 
    scale: float = 30.0, 
    show: bool = True, 
    H_OdEn: Optional[np.ndarray] = None, 
    result: Optional[Dict[str, Any]] = None, 
    save: bool = False, 
    save_path: Optional[str] = None
) -> go.Figure:
    """
    여러 좌표계 프레임을 시각화
    H_dict: {이름: 4x4 행렬}
    H_OdEn: 엔드이펙터 기준 오브젝트 HM
    """
    traces = []

    if result and H_OdEn is not None:
        mesh_sil = result["mesh_quad"]
        # H_OdEn 으로 mesh 회전 후 fig에 추가
        mesh_tf = mesh_sil.copy()
        mesh_tf.apply_transform(H_OdEn)
        x, y, z = mesh_tf.vertices.T
        i, j, k = mesh_tf.faces.T
        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color="#cfcfcf", opacity=0.3,
            name="object_mesh", legendgroup="object_mesh", showlegend=True
        ))

    for i, (name, H) in enumerate(H_dict.items()):
        traces += frame_traces(H, name, scale=scale, legendgroup=name, showlegend=True)

    fig = go.Figure(traces)
    fig.update_layout(
        title="Coordinate Frames Visualization",
        margin=dict(l=0,r=0,t=30,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # 그룹 단위 토글
        scene=dict(
            xaxis=dict(showgrid=True, zeroline=True, range=[-1000, 1000]),
            yaxis=dict(showgrid=True, zeroline=True, range=[-1000, 1000]),
            zaxis=dict(showgrid=True, zeroline=True, range=[-1000, 1000]),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    if save and save_path:
        fig.write_html(save_path)
    return fig


# 점 마찰 GWS 시각화용
def _hull_to_mesh3d(hull: ConvexHull, color: str, opacity: float, name: str) -> go.Mesh3d:
    """ scipy.spatial.ConvexHull 객체를 Plotly Mesh3d로 변환합니다. """
    x, y, z = hull.points.T
    i, j, k = hull.simplices.T # hull.simplices가 표면 삼각형을 정의
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, flatshading=True,
        name=name, showlegend=True,
        lighting=dict(ambient=0.4, diffuse=1.0, specular=0.1)
    )

def visualize_wrench_space(
    c_i: np.ndarray, 
    n_i: np.ndarray, 
    c_j: np.ndarray, 
    n_j: np.ndarray, 
    com: np.ndarray, 
    mu: float = gf.PadParams.mu, 
    k: int = 8, 
    show_force: bool = True, 
    show_torque: bool = True, 
    show: bool = False
) -> go.Figure:
    """
    Grasp Wrench Space (GWS)의 3D 투영인 Force Space와 Torque Space를 시각화
    """
    
    # 1. 렌치 계산
    # gf.unit 사용
    inward_n_i = -gf.unit(np.asarray(n_i))
    inward_n_j = -gf.unit(np.asarray(n_j))
    
    # gf.get_friction_cone_vectors 사용
    cone_vecs_i = gf.get_friction_cone_vectors(inward_n_i, mu, k)
    cone_vecs_j = gf.get_friction_cone_vectors(inward_n_j, mu, k)
    
    r_i = np.asarray(c_i) - np.asarray(com)
    r_j = np.asarray(c_j) - np.asarray(com)
    
    wrenches = []
    for f_i in cone_vecs_i:
        wrenches.append(np.concatenate([f_i, np.cross(r_i, f_i)]))
    for f_j in cone_vecs_j:
        wrenches.append(np.concatenate([f_j, np.cross(r_j, f_j)]))
        
    wrenches = np.array(wrenches)
    
    traces = []
    
    # 2. Force Space (TWS) 시각화
    if show_force:
        forces = wrenches[:, :3]
        try:
            hull_force = ConvexHull(forces)
            traces.append(_hull_to_mesh3d(hull_force, 
                          color="black", opacity=0.3, 
                          name="Force Space (TWS)"))
        except Exception as e:
            print(f"Could not compute 3D Force Hull: {e}. Plotting points.")
            traces.append(go.Scatter3d(x=forces[:,0], y=forces[:,1], z=forces[:,2], 
                          mode='markers', marker=dict(color="#0092BB", size=2), 
                          name="Force Vectors (primitive)"))

    # 3. Torque Space (RWS) 시각화
    if show_torque:
        torques = wrenches[:, 3:]
        try:
            hull_torque = ConvexHull(torques)
            traces.append(_hull_to_mesh3d(hull_torque, 
                          color="black", opacity=0.3, 
                          name="Torque Space (RWS)"))
        except Exception as e:
            print(f"Could not compute 3D Torque Hull: {e}. Plotting points.")
            traces.append(go.Scatter3d(x=torques[:,0], y=torques[:,1], z=torques[:,2], 
                          mode='markers', marker=dict(color="#0092BB", size=2), 
                          name="Torque Vectors (primitive)"))
    
    # 4. Plotly Figure 생성
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Wrench Space Projections (mu={mu}, k={k})",
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(x=0),
        scene=dict(
            xaxis_title="F_x / T_x (N / N-mm)",
            yaxis_title="F_y / T_y (N / N-mm)",
            zaxis_title="F_z / T_z (N / N-mm)",
            aspectmode="data"  # 축 스케일을 동일하게 유지하지 않고 데이터에 맞춤
        )
    )
    if show:
        fig.show(renderer="browser")
    return fig


# 면 마찰 GWS 시각화용
def visualize_squeeze_wrench_space(
    mesh: trimesh.Trimesh,
    f_i_idx: Union[List[int], np.ndarray], f_j_idx: Union[List[int], np.ndarray],
    c_i: np.ndarray, n_i: np.ndarray, yaw_i: float,
    c_j: np.ndarray, n_j: np.ndarray, yaw_j: float,
    com: np.ndarray,
    pad_w: float = gf.PadParams.pad_w,
    pad_h: float = gf.PadParams.pad_h,
    pad_d: float = gf.PadParams.pad_d,
    squeeze_depth: float = 0.1,
    mu: float = gf.PadParams.mu,
    k: int = 8,
    show: bool = True
) -> Optional[go.Figure]:
    """
    Squeeze 모델을 사용하여 추출된 접촉점들과,
    그로 인해 형성되는 Force Space(TWS) 및 Torque Space(RWS)를 시각화합니다.
    """
    
    # 1. 접촉점 및 패드 박스 추출 (Squeeze 로직 사용)
    # 최적화를 위해 부분 메쉬 사용
    # gf.get_points_in_squeezed_pad 사용
    mesh_i = mesh.submesh([f_i_idx], append=True)
    pts_i, box_i = gf.get_points_in_squeezed_pad(mesh_i, {"centroid": c_i, "normal": n_i}, 
                                                 pad_w, pad_h, pad_d, yaw_i, squeeze_depth)
    
    mesh_j = mesh.submesh([f_j_idx], append=True)
    pts_j, box_j = gf.get_points_in_squeezed_pad(mesh_j, {"centroid": c_j, "normal": n_j}, 
                                                 pad_w, pad_h, pad_d, yaw_j, squeeze_depth)

    all_points = []
    if len(pts_i) > 0: all_points.append(pts_i)
    if len(pts_j) > 0: all_points.append(pts_j)
    
    if not all_points:
        print("No contact points found with squeeze model.")
        return None

    contact_points = np.vstack(all_points)
    
    # 2. Wrench 계산 (Force & Torque)
    forces = []
    torques = []
    
    def collect_wrenches(points, normal):
        # gf.basis_from_normal 사용
        t, b, _ = gf.basis_from_normal(normal)
        angle_step = 2 * np.pi / k
        for pt in points:
            r = pt - com
            for i in range(k):
                theta = i * angle_step
                # Local Force Vector (Friction Cone Boundary)
                f = normal + mu * np.cos(theta) * t + mu * np.sin(theta) * b
                f = f / np.linalg.norm(f) # Normalize
                
                tau = np.cross(r, f)

                forces.append(f)
                torques.append(tau)

    collect_wrenches(pts_i, -n_i)
    collect_wrenches(pts_j, -n_j)
    
    forces = np.array(forces)
    torques = np.array(torques)
    
    # 3. 시각화 구성 (1행 3열 서브플롯)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Spatial Contact View", "Force Space (TWS)", "Torque Space (RWS)")
    )
    
    # --- Scene 1: Spatial View (Mesh + Pads + Contact Points) ---
    # A. Mesh (전체 메쉬, 반투명)
    x, y, z = mesh.vertices.T
    I, J, K = mesh.faces.T
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, 
                            color='#cfcfcf', opacity=0.6, name='Object Mesh'), row=1, col=1)
    
    # B. Squeezed Pads (Visualizing the intrusion)
    for box, name, color in [(box_i, 'Pad I', "#0092BB"), (box_j, 'Pad J', "#0092BB")]:
        bx, by, bz = box.vertices.T
        bI, bJ, bK = box.faces.T
        fig.add_trace(go.Mesh3d(x=bx, y=by, z=bz, i=bI, j=bJ, k=bK, 
                                color=color, opacity=0.2, name=name), row=1, col=1)
        
    # C. Contact Points (실제 힘이 가해지는 점들)
    fig.add_trace(go.Scatter3d(
        x=contact_points[:,0], y=contact_points[:,1], z=contact_points[:,2],
        mode='markers', marker=dict(size=2, color="#0092BB"), name='Contact Points'
    ), row=1, col=1)

    # D. CoM
    fig.add_trace(go.Scatter3d(
        x=[com[0]], y=[com[1]], z=[com[2]],
        mode='markers', marker=dict(size=5, color='black', symbol='diamond',opacity=0.5), name='CoM'
    ), row=1, col=1)

    # --- Scene 2: Force Space (Convex Hull) ---
    try:
        hull_f = ConvexHull(forces)
        # Hull Vertices
        fig.add_trace(go.Scatter3d(
            x=forces[:,0], y=forces[:,1], z=forces[:,2],
            mode='markers', marker=dict(size=2, color="#0092BB", opacity=0.5), name='Force Vectors EndPoints'
        ), row=1, col=2)
        # Hull Surface
        hf = hull_f.simplices
        fig.add_trace(go.Mesh3d(
            x=forces[:,0], y=forces[:,1], z=forces[:,2],
            i=hf[:,0], j=hf[:,1], k=hf[:,2],
            color="#0092BB", opacity=0.3, name='Force Hull'
        ), row=1, col=2)
        # Origin
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='black'), name='Origin'), row=1, col=2)
        
    except Exception as e:
        print(f"Force Hull Error: {e}")

    # --- Scene 3: Torque Space (Convex Hull) ---
    try:
        hull_t = ConvexHull(torques)
        # Hull Vertices
        fig.add_trace(go.Scatter3d(
            x=torques[:,0], y=torques[:,1], z=torques[:,2],
            mode='markers', marker=dict(size=2, color="#0092BB", opacity=0.5), name='Torque Vectors EndPoints'
        ), row=1, col=3)
        # Hull Surface
        ht = hull_t.simplices
        fig.add_trace(go.Mesh3d(
            x=torques[:,0], y=torques[:,1], z=torques[:,2],
            i=ht[:,0], j=ht[:,1], k=ht[:,2],
            color="#0092BB", opacity=0.3, name='Torque Hull'
        ), row=1, col=3)
        # Origin
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='black'), name='Origin'), row=1, col=3)

    except Exception as e:
        print(f"Torque Hull Error: {e}")

    # Layout 설정
    fig.update_layout(
        title="Squeeze Grasp Wrench Space (Surface Contact Model)",
        margin=dict(r=0, l=0, b=0, t=40),
        
        # Scene 설정 수정
        scene1=dict(
            aspectmode='data',
            xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')
        ),
        scene2=dict(
            aspectmode='data',
            xaxis=dict(title='Fx'), yaxis=dict(title='Fy'), zaxis=dict(title='Fz')
        ),
        scene3=dict(
            aspectmode='data',
            xaxis=dict(title='Tx'), yaxis=dict(title='Ty'), zaxis=dict(title='Tz')
        ),
    )

    if show:
        fig.show(renderer="browser")
        
    return fig
