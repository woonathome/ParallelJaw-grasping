import numpy as np
import plotly.graph_objects as go
import trimesh
from dataclasses import dataclass

# -------------------------------
# helpers
# -------------------------------
@dataclass
class PadParams:
    pad_w: float = 34.0 # 34
    pad_h: float = 21.0 # 21
    pad_d: float = 7.0 # 7

def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def mesh3d_from_trimesh(m, face_indices=None, color="lightgray", opacity=0.3, name="mesh"):
    faces = m.faces if face_indices is None else m.faces[np.asarray(list(face_indices), dtype=int)]
    x, y, z = m.vertices.T
    i, j, k = faces.T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.2, roughness=0.9),
        lightposition=dict(x=0, y=0, z=1),
        name=name, showlegend=True
    )

def boundary_edges_trace(mesh, face_idx, color="#166534", width=1, name="boundary"):
    """patch 경계선(외곽 edge만) Scatter3d로 표현"""
    F = mesh.faces[np.asarray(list(face_idx), dtype=int)]
    # edge -> count
    edges = {}
    for a,b,c in F:
        for u,v in ((a,b),(b,c),(c,a)):
            e = (u,v) if u < v else (v,u)
            edges[e] = edges.get(e, 0) + 1
    # 경계(edge count==1)만 라인화
    lines = []
    for (u,v), c in edges.items():
        if c == 1:
            p1, p2 = mesh.vertices[u], mesh.vertices[v]
            lines.extend([p1, p2, [None, None, None]])  # 분리용 None
    if not lines:
        return None
    L = np.array(lines, dtype=float)
    return go.Scatter3d(
        x=L[:,0], y=L[:,1], z=L[:,2],
        mode="lines", line=dict(color=color, width=width),
        name=name, showlegend=False
    )

def hsv(i, n, s=0.6, v=0.95):
    import colorsys
    h = (i % n) / max(1, n)
    r,g,b = colorsys.hsv_to_rgb(h, s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def cones_for_normals(
    patches,
    mesh_bounds,
    normal_scale: float = 20.0,   # ↑ 키울수록 cone 커짐
    min_len_ratio: float = 0.1, # bbox 대각선 대비 최소 벡터 길이
    color: str = "#000000",
    name: str = "normals",
):
    """
    - 각 패치 길이 nlen = max(bbox_min_len, 0.5*sqrt(area))
    - sizeref = median(nlen) / normal_scale  (outlier 영향 완화)
    """
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

def basis_from_normal(n):
    n=unit(n); a=np.array([1,0,0]) if abs(n[0])<0.9 else np.array([0,1,0])
    u=unit(np.cross(n,a)); v=np.cross(n,u)
    return u,v,n

def rotate_byaxis(axis, ang_deg):
    axis = unit(axis)
    th = np.deg2rad(ang_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], float)
    I = np.eye(3)
    return I + np.sin(th)*K + (1-np.cos(th))*(K@K)

def make_pad_box_at_patch(patch, pad_w, pad_h, pad_d):
    """
    패치 평면의 바깥쪽(+n)으로 'pad_d' 만큼 돌출된 OBB 생성
    + centroid에서 바깥쪽으로 lift_mm 만큼 추가 이격
    - extents: [pad_w, pad_h, pad_d]
    - center : 패치 중심 + n * (extents_z/2)
    - orientation: columns = [u, v, n]
    """
    n = unit(np.asarray(patch["normal"], float))
    u, v, n = basis_from_normal(n)
    c = np.asarray(patch["centroid"], float)

    ext_z = float(pad_d)
    box = trimesh.creation.box(extents=[float(pad_w), float(pad_h), float(pad_d)])

    T = np.eye(4)
    T[:3, :3] = np.column_stack([u, v, n])
    T[:3,  3] = c + n*0.5*ext_z
    box.apply_transform(T)

    return box

def make_pad_box_at_patch_with_yaw(patch, pad_w, pad_h, pad_d, yaw_deg):
    """
    패치 평면의 바깥쪽(+n)으로 'pad_d' 만큼 돌출된 OBB 생성
    + centroid에서 바깥쪽으로 lift_mm 만큼 추가 이격
    - extents: [pad_w, pad_h, pad_d]
    - center : 패치 중심 + n * (extents_z/2)
    - orientation: columns = [u, v, n]
    """
    n = unit(np.asarray(patch["normal"], float))
    u, v, n = basis_from_normal(n)
    c = np.asarray(patch["centroid"], float)

    ext_z = float(pad_d)
    box = trimesh.creation.box(extents=[float(pad_w), float(pad_h), float(pad_d)])

    # 로컬 yaw 회전: R_world = B @ Rz(yaw)
    B = np.column_stack([u, v, n])
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], float)
    R_world = B @ Rz
    n_world = R_world[:, 2]
    T = np.eye(4)
    T[:3, :3] = R_world
    T[:3,  3] = c + n_world*0.5*ext_z
    box.apply_transform(T)

    return box

# -------------------------------
# 1) 추출된 패치 시각화
# -------------------------------
def visualize_merged_patches_plotly(result,
                                    show_silhouette=True,
                                    show=False):
    """
    result: grasping.compute_best_patch_pairs(...) 출력 dict
      - mesh_quad: 원본 실루엣용 Trimesh
      - mesh_patches: 패치가 추출된(간소화) Trimesh
      - patches: [{'id','face_indices','centroid','normal','b','area'}, ...]
    """
    mesh_sil = result["mesh_quad"]
    mesh_p   = result["mesh_patches"]
    patches  = result["patches"]

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
    traces.append(cones_for_normals(patches, mesh_bounds=mesh_sil.bounds, normal_scale=100.0,
                                     min_len_ratio=0.1, color=normal_color, name="normals"))
    
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
        fig.show()

    return fig

# -------------------------------
# 2) patch pair(top_k) 시각화
# -------------------------------
def visualize_pairs_centroid_lines(result,
                                   show_silhouette=True,
                                   patch_opacity=0.3,
                                   show=False):
    """
    result['top_k']의 모든 pair를 표시:
      - pair마다 두 패치(같은 색, 옅게)
      - 두 패치의 centroid를 잇는 선
    """
    mesh_sil = result["mesh_quad"]
    mesh_p = result["mesh_patches"]
    patches = result["patches"]
    cands = result.get("top_k", [])
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
        fig.show()

    return fig

# -------------------------------
# 3) feasible patch pair 시각화
# -------------------------------
def visualize_feasible_pairs_pads(result, reports,
                                  show_silhouette=True,
                                  pad_w=PadParams.pad_w,
                                  pad_h=PadParams.pad_h,
                                  pad_d=PadParams.pad_d,
                                  show=False):
    mesh = result.get("mesh_quad", result.get("mesh_patches"))
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    # print(ok_idx)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.8, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        yaw_deg = reports[k]['feasible_yaw']
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]
        col = hsv(k, len(ok_idx), s=0.55, v=0.95)

        # # pad boxes
        # try:
        #     box_i = make_pad_box_at_patch(pi, pad_w, pad_h, pad_d)
        #     box_j = make_pad_box_at_patch(pj, pad_w, pad_h, pad_d)
        # except NameError:
        #     box_i = make_pad_box_at_patch(pi, pad_w, pad_h, pad_d)
        #     box_j = make_pad_box_at_patch(pj, pad_w, pad_h, pad_d)

        # pad boxes + rot
        try:
            box_i = make_pad_box_at_patch_with_yaw(pi, pad_w, pad_h, pad_d,  yaw_deg)
            box_j = make_pad_box_at_patch_with_yaw(pj, pad_w, pad_h, pad_d, -yaw_deg)
        except NameError:
            box_i = make_pad_box_at_patch_with_yaw(pi, pad_w, pad_h, pad_d,  yaw_deg)
            box_j = make_pad_box_at_patch_with_yaw(pj, pad_w, pad_h, pad_d, -yaw_deg)

        # legendgroup으로 두 pad를 하나의 토글 그룹에 묶기
        lg = f"pair {idx}"
        # proxy(범례 핸들) – 클릭 시 그룹 전체 토글
        ci = np.asarray(pi["centroid"], float); cj = np.asarray(pj["centroid"], float)
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True
        ))
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
        fig.show()

    return fig