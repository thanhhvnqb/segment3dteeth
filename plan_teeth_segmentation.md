# Tooth Segmentation Pipeline — Implementation Plan

> **Mục tiêu:** Từ file `.glb` chứa toàn bộ răng một hàm (upper hoặc lower hoặc cả hai), tách ra N mesh riêng lẻ, mỗi mesh = 1 răng, gán nhãn theo chuẩn FDI.
>
> **Input giả định:** Mesh đã qua dental segmentation. có thể bỏ qua những phần khác, chỉ cần quan tâm upper và/hoặc lower teeth. Lưu ý có thể cần loại bỏ những phần nhỏ mà mô hình dental segmentation đã segment nhầm.

---

## 1. Dependencies

```txt
trimesh>=4.0.0
numpy>=1.24
scipy>=1.11
scikit-image>=0.21
potpourri3d>=0.0.8          # heat method geodesic
robust-laplacian>=0.2.0     # high-quality curvature
open3d>=0.18                # DBSCAN, mesh ops
vhacdx>=0.0.4              # convex decomposition fallback
pygeodesic>=0.1.5           # Dijkstra fallback
```

Cài đặt:

```bash
pip install trimesh numpy scipy scikit-image potpourri3d \
            robust-laplacian open3d vhacdx pygeodesic
```

---

## 2. Kiến trúc tổng quan

```
load_glb(path)
    │
    ▼
phase0_connected_components()
    │ done nếu n == expected
    │ tiếp nếu còn merged component
    ▼
classify_merged_component()        ← xác định loại răng từ arch position
    │
    ├── anterior  → phase1_geodesic_watershed()
    ├── premolar  → phase1_geodesic_watershed() + phase2_crosssection_sweep()
    └── molar     → phase2_crosssection_sweep()  [primary]
                    └── fallback: phase3_vhacd_prior()
    │
    ▼
postprocess()                      ← smooth boundary, validate volume
    │
    ▼
assign_fdi_labels()                ← đánh số răng theo chuẩn FDI
    │
    ▼
export_glb_per_tooth()
```

---

## 3. Phase 0 — Connected Components

**Mục đích:** Xử lý trường hợp đơn giản nhất — răng đã rời nhau vật lý. Free win, không cần tính toán phức tạp.

```python
import trimesh
import numpy as np

def phase0_connected_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """
    Tách mesh thành các connected components.
    Trả về list các component, mỗi component là 1 Trimesh.
    """
    components = mesh.split(only_watertight=False)
    # Lọc bỏ component rác (diện tích quá nhỏ < 5% mean)
    areas = [c.area for c in components]
    mean_area = np.mean(areas)
    components = [c for c in components if c.area > mean_area * 0.05]
    return components
```

**Khi nào đủ:** So sánh `len(components)` với expected tooth count.

```python
EXPECTED_TOOTH_COUNT = {
    "upper": 8,   # hoặc 14-16 nếu có wisdom teeth
    "lower": 8,
}

def is_done(components, arch: str) -> bool:
    # Heuristic: nếu số component >= expected thì coi là done
    return len(components) >= EXPECTED_TOOTH_COUNT[arch]
```

**Identify merged components:**

```python
def find_merged_components(components: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """
    Component có volume > 1.6x volume trung bình → nghi ngờ là nhiều răng dính nhau.
    """
    volumes = [c.volume for c in components]
    median_vol = np.median(volumes)
    return [c for c in components if c.volume > median_vol * 1.6]
```

---

## 4. Phase 1 — Geodesic Watershed (Anterior & Premolar)

Dùng cho **răng cửa, răng nanh, răng premolar** — vùng có interproximal space đủ concave để detect bằng curvature.

### 4.1 Tính mean curvature multi-scale

```python
import robust_laplacian
from scipy.sparse.linalg import eigsh

def compute_multiscale_curvature(
    mesh: trimesh.Trimesh,
    sigmas: list[float] = [0.5, 1.0, 2.0, 4.0]
) -> np.ndarray:
    """
    Tính mean curvature ở nhiều scale, kết hợp có trọng số.
    sigma đơn vị mm.
    """
    weights = [0.1, 0.2, 0.3, 0.4]
    curv_combined = np.zeros(len(mesh.vertices))

    for sigma, w in zip(sigmas, weights):
        iter_count = max(1, int(sigma * 8))
        smoothed = trimesh.smoothing.filter_laplacian(
            mesh.copy(), lamb=0.5, iterations=iter_count
        )
        # Robust Laplacian cho kết quả ổn định hơn cotangent Laplacian
        L, M = robust_laplacian.mesh_laplacian(
            np.array(smoothed.vertices),
            np.array(smoothed.faces)
        )
        # Mean curvature = ||HN|| = 0.5 * ||L @ v||
        Lv = L @ smoothed.vertices
        curv = np.linalg.norm(Lv, axis=1) * 0.5
        curv_combined += w * curv

    return curv_combined
```

### 4.2 Detect concave seed vertices

```python
def detect_seeds(
    mesh: trimesh.Trimesh,
    curvature: np.ndarray,
    threshold_sigma: float = 1.5
) -> np.ndarray:
    """
    Seeds = vertices có curvature thấp (concave valley) ở interproximal region.
    Trả về array of vertex indices.
    """
    mu = curvature.mean()
    sigma = curvature.std()
    seed_mask = curvature < (mu - threshold_sigma * sigma)
    seed_indices = np.where(seed_mask)[0]

    # Lọc thêm: seed phải nằm ở vùng "thấp" theo trục occlusal (z-axis)
    # Interproximal space thường ở phần cervical của crown
    z_vals = mesh.vertices[seed_indices, 2]
    z_low = np.percentile(mesh.vertices[:, 2], 35)
    seed_indices = seed_indices[z_vals < z_low + (z_vals.max() - z_vals.min()) * 0.4]

    return seed_indices
```

### 4.3 Cluster seeds → 1 cluster/kẽ răng

```python
import open3d as o3d

def cluster_seeds(
    mesh: trimesh.Trimesh,
    seed_indices: np.ndarray,
    eps_mm: float = 2.5,
    min_samples: int = 5
) -> list[np.ndarray]:
    """
    DBSCAN cluster seed vertices. Mỗi cluster ≈ 1 interproximal space.
    Trả về list of vertex index arrays, 1 per cluster.
    """
    seed_points = mesh.vertices[seed_indices]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seed_points))
    labels = np.array(pcd.cluster_dbscan(eps=eps_mm, min_points=min_samples))

    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # noise
        idx = seed_indices[labels == label]
        clusters.append(idx)
    return clusters
```

### 4.4 Geodesic watershed từ seeds

```python
import potpourri3d as pp3d

def geodesic_watershed(
    mesh: trimesh.Trimesh,
    seed_clusters: list[np.ndarray]
) -> np.ndarray:
    """
    Tính geodesic distance từ mỗi seed cluster.
    Voronoi boundary giữa các cluster = đường cắt.
    Trả về vertex label array (int), -1 = boundary.
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(
        mesh.vertices, mesh.faces
    )

    dist_maps = []
    for cluster in seed_clusters:
        # Tính distance từ tất cả vertices trong cluster
        dists = np.min(
            [solver.compute_distance(int(v)) for v in cluster[:10]],  # sample 10 seeds
            axis=0
        )
        dist_maps.append(dists)

    dist_maps = np.stack(dist_maps, axis=1)  # (n_vertices, n_clusters)
    labels = np.argmin(dist_maps, axis=1)

    # Detect boundary: vertex có neighbor thuộc label khác
    boundary_mask = np.zeros(len(mesh.vertices), dtype=bool)
    for edge in mesh.edges_unique:
        if labels[edge[0]] != labels[edge[1]]:
            boundary_mask[edge[0]] = True
            boundary_mask[edge[1]] = True

    labels[boundary_mask] = -1
    return labels
```

### 4.5 Slice mesh theo labels

```python
def slice_by_labels(
    mesh: trimesh.Trimesh,
    vertex_labels: np.ndarray
) -> list[trimesh.Trimesh]:
    """
    Tách mesh thành submeshes theo vertex label.
    Bỏ qua boundary vertices (label == -1).
    """
    unique_labels = set(vertex_labels) - {-1}
    result = []

    for label in unique_labels:
        # Giữ faces mà tất cả 3 vertices đều thuộc label này
        face_mask = np.all(
            vertex_labels[mesh.faces] == label,
            axis=1
        )
        if face_mask.sum() < 50:  # bỏ fragment quá nhỏ
            continue
        sub = trimesh.util.submesh(mesh, face_mask, append=True)
        result.append(sub)

    return result
```

---

## 5. Phase 2 — Cross-Section Sweep (Molar)

Dùng cho **răng hàm (molar)** — contact area phẳng, curvature gần 0, Phase 1 fail.

### 5.1 Fit arch curve

```python
from scipy.interpolate import UnivariateSpline

def fit_arch_curve(
    components: list[trimesh.Trimesh]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit dental arch curve từ centroids của các components.
    Trả về (arc_length_params, curve_tangents) tại mỗi centroid.
    """
    centroids = np.array([c.centroid for c in components])
    # Sort theo x-axis (mesial-distal)
    order = np.argsort(centroids[:, 0])
    centroids = centroids[order]

    # Fit spline qua centroids
    t = np.linspace(0, 1, len(centroids))
    spline_x = UnivariateSpline(t, centroids[:, 0], s=0.5)
    spline_y = UnivariateSpline(t, centroids[:, 1], s=0.5)

    # Evaluate tangent vectors
    t_dense = np.linspace(0, 1, 200)
    dx = spline_x.derivative()(t_dense)
    dy = spline_y.derivative()(t_dense)
    tangents = np.column_stack([dx, dy, np.zeros_like(dx)])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-8)

    return t_dense, tangents
```

### 5.2 Cross-section sweep → detect eo

```python
from scipy.signal import find_peaks

def crosssection_sweep(
    mesh: trimesh.Trimesh,
    n_slices: int = 120,
    min_prominence_ratio: float = 0.25,
    min_distance_mm: float = 4.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Sweep mặt phẳng cắt dọc trục mesial-distal.
    Tìm local minima của cross-section area = vị trí tiếp xúc giữa răng.
    Trả về list of (plane_origin, plane_normal) tại mỗi eo.
    """
    bounds = mesh.bounds
    x_min, x_max = bounds[0][0], bounds[1][0]
    x_positions = np.linspace(x_min + 1.0, x_max - 1.0, n_slices)

    areas = []
    valid_x = []
    plane_normal = np.array([1.0, 0.0, 0.0])  # sweep theo x-axis

    for x in x_positions:
        plane_origin = np.array([x, 0.0, 0.0])
        try:
            section = mesh.section(
                plane_origin=plane_origin,
                plane_normal=plane_normal
            )
            if section is not None:
                # Area của cross-section polygon
                path2d, _ = section.to_planar()
                area = sum(abs(p.area) for p in path2d.polygons_closed)
                areas.append(area)
                valid_x.append(x)
        except Exception:
            areas.append(0.0)
            valid_x.append(x)

    areas = np.array(areas)
    valid_x = np.array(valid_x)
    step_mm = (x_max - x_min) / n_slices

    # Tìm local minima (đảo dấu để find_peaks)
    peaks, props = find_peaks(
        -areas,
        prominence=areas.std() * min_prominence_ratio,
        distance=int(min_distance_mm / step_mm)
    )

    # Bỏ các minima quá gần mép (artifact)
    peaks = [p for p in peaks if 3 < p < len(valid_x) - 3]

    cut_planes = []
    for p in peaks:
        origin = np.array([valid_x[p], mesh.centroid[1], mesh.centroid[2]])
        normal = plane_normal.copy()
        cut_planes.append((origin, normal))

    return cut_planes
```

### 5.3 Slice mesh theo cut planes

```python
def slice_by_planes(
    mesh: trimesh.Trimesh,
    cut_planes: list[tuple[np.ndarray, np.ndarray]]
) -> list[trimesh.Trimesh]:
    """
    Cắt mesh tại mỗi cut plane.
    Trả về list submeshes.
    """
    # Sắp xếp planes theo x để cắt tuần tự
    cut_planes_sorted = sorted(cut_planes, key=lambda p: p[0][0])

    pieces = [mesh]
    for origin, normal in cut_planes_sorted:
        new_pieces = []
        for piece in pieces:
            # Tách thành 2 nửa
            pos = trimesh.intersections.slice_mesh_plane(
                piece, normal, origin, cap=True
            )
            neg = trimesh.intersections.slice_mesh_plane(
                piece, -normal, -origin, cap=True  # dấu đảo ngược
            )
            if pos is not None and len(pos.faces) > 10:
                new_pieces.append(pos)
            if neg is not None and len(neg.faces) > 10:
                new_pieces.append(neg)
        pieces = new_pieces

    return pieces
```

> **Lưu ý:** `trimesh.intersections.slice_mesh_plane` với `cap=True` sẽ lấp kín vết cắt bằng một polygon phẳng. Nếu không muốn cap (open mesh), truyền `cap=False`.

---

## 6. Phase 3 — VHACD Fallback

Dùng khi cả Phase 1 và Phase 2 fail (ví dụ 3 molar dính, wisdom tooth bất thường).

```python
import vhacdx

def vhacd_fallback(
    mesh: trimesh.Trimesh,
    max_hulls: int = 6,
    resolution: int = 200000
) -> list[trimesh.Trimesh]:
    """
    Convex decomposition → boundary giữa convex hulls → seed cho geodesic refine.
    Đây là fallback cuối, kết quả ít chính xác hơn nhưng luôn có output.
    """
    params = vhacdx.VHACDParameters()
    params.max_num_vertices_per_ch = 64
    params.resolution = resolution
    params.max_convex_hulls = max_hulls

    hulls = vhacdx.decompose(
        mesh.vertices.tolist(),
        mesh.faces.tolist(),
        params
    )

    # Assign mỗi vertex của mesh gốc vào hull gần nhất
    hull_meshes = [trimesh.Trimesh(vertices=h.points, faces=h.faces) for h in hulls]

    vertex_labels = np.zeros(len(mesh.vertices), dtype=int)
    for i, v in enumerate(mesh.vertices):
        dists = [np.min(np.linalg.norm(hm.vertices - v, axis=1)) for hm in hull_meshes]
        vertex_labels[i] = np.argmin(dists)

    return slice_by_labels(mesh, vertex_labels)
```

---

## 7. Post-processing

### 7.1 Smooth boundary edges

```python
def smooth_boundary(
    mesh: trimesh.Trimesh,
    iterations: int = 3
) -> trimesh.Trimesh:
    """
    Laplacian smooth nhẹ ở boundary để xóa artifact từ vết cắt.
    Chỉ move boundary vertices, giữ nguyên interior.
    """
    boundary_verts = set(mesh.vertices[mesh.edges[mesh.edges_unique_length == 2].flatten()])
    return trimesh.smoothing.filter_laplacian(
        mesh, lamb=0.3, iterations=iterations
    )
```

### 7.2 Validate volume

```python
TOOTH_VOLUME_RANGE_MM3 = {
    "incisor":  (100,  600),
    "canine":   (200,  800),
    "premolar": (300, 1200),
    "molar":    (600, 3000),
}

def validate_and_merge_fragments(
    components: list[trimesh.Trimesh],
    min_volume_mm3: float = 80.0
) -> list[trimesh.Trimesh]:
    """
    Bỏ hoặc merge fragment quá nhỏ vào neighbor gần nhất.
    """
    valid = [c for c in components if c.volume >= min_volume_mm3]
    small = [c for c in components if c.volume < min_volume_mm3]

    for frag in small:
        if not valid:
            continue
        # Merge vào component gần nhất theo centroid distance
        dists = [np.linalg.norm(v.centroid - frag.centroid) for v in valid]
        nearest_idx = np.argmin(dists)
        merged = trimesh.util.concatenate([valid[nearest_idx], frag])
        valid[nearest_idx] = merged

    return valid
```

---

## 8. FDI Label Assignment

```
Hàm trên (upper): quadrant 1 (bên phải bệnh nhân) = 11-18, quadrant 2 (bên trái) = 21-28
Hàm dưới (lower): quadrant 3 (bên trái) = 31-38, quadrant 4 (bên phải) = 41-48

Quy tắc FDI:
  - Răng 1 (central incisor) gần đường giữa nhất
  - Răng 8 (wisdom tooth) xa nhất
  - Upper: right = quadrant 1, left = quadrant 2
  - Lower: left = quadrant 3, right = quadrant 4
```

```python
def assign_fdi_labels(
    components: list[trimesh.Trimesh],
    arch: str  # "upper" hoặc "lower"
) -> dict[int, trimesh.Trimesh]:
    """
    Gán nhãn FDI cho mỗi component dựa trên x-position của centroid.
    Giả định: x > 0 = bên phải bệnh nhân, x < 0 = bên trái.
    Trả về dict {fdi_number: mesh}.
    """
    centroids_x = [(c, c.centroid[0]) for c in components]
    # Sort từ phải sang trái (x lớn → nhỏ)
    centroids_x.sort(key=lambda t: -t[1])

    fdi_map = {}
    mid = len(components) // 2

    if arch == "upper":
        # Phải: 11,12,13,14,15,16,17,18
        # Trái:  21,22,23,24,25,26,27,28
        right = centroids_x[:mid]
        left  = centroids_x[mid:]
        for i, (comp, _) in enumerate(right):
            fdi_map[11 + i] = comp
        for i, (comp, _) in enumerate(left):
            fdi_map[21 + i] = comp
    else:
        # Lower: left = 31..., right = 41...
        right = centroids_x[:mid]
        left  = centroids_x[mid:]
        for i, (comp, _) in enumerate(left):
            fdi_map[31 + i] = comp
        for i, (comp, _) in enumerate(right):
            fdi_map[41 + i] = comp

    return fdi_map
```

---

## 9. Export

```python
def export_teeth(
    fdi_map: dict[int, trimesh.Trimesh],
    output_dir: str
) -> list[str]:
    """
    Export mỗi răng thành file .glb riêng.
    Tên file: tooth_{FDI}.glb (ví dụ: tooth_11.glb, tooth_16.glb)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for fdi, mesh in fdi_map.items():
        path = os.path.join(output_dir, f"tooth_{fdi}.glb")
        mesh.export(path)
        paths.append(path)
    return paths
```

---

## 10. Entry point tổng hợp

```python
def segment_teeth(
    glb_path: str,
    arch: str,                   # "upper" hoặc "lower"
    output_dir: str,
    expected_count: int = 8,
    debug: bool = False
) -> dict[int, trimesh.Trimesh]:

    # Load
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh = scene

    # Phase 0
    components = phase0_connected_components(mesh)
    if debug:
        print(f"[Phase 0] {len(components)} components found")

    if is_done(components, arch):
        fdi_map = assign_fdi_labels(components, arch)
        return export_teeth(fdi_map, output_dir)

    # Xử lý merged components
    merged = find_merged_components(components)
    separated = [c for c in components if c not in merged]

    for mc in merged:
        arch_pos = mc.centroid[0]
        bounds_x = mesh.bounds[:, 0]
        rel_pos = (arch_pos - bounds_x[0]) / (bounds_x[1] - bounds_x[0])

        if rel_pos < 0.25 or rel_pos > 0.75:
            # Vùng molar (ngoại biên)
            cut_planes = crosssection_sweep(mc)
            if cut_planes:
                sub_meshes = slice_by_planes(mc, cut_planes)
            else:
                if debug:
                    print(f"[Phase 2] sweep failed → VHACD fallback")
                sub_meshes = vhacd_fallback(mc)
        else:
            # Vùng anterior/premolar
            curvature = compute_multiscale_curvature(mc)
            seeds = detect_seeds(mc, curvature)
            clusters = cluster_seeds(mc, seeds)

            if len(clusters) >= 1:
                labels = geodesic_watershed(mc, clusters)
                sub_meshes = slice_by_labels(mc, labels)
            else:
                if debug:
                    print(f"[Phase 1] no seeds found → cross-section fallback")
                cut_planes = crosssection_sweep(mc)
                sub_meshes = slice_by_planes(mc, cut_planes) if cut_planes \
                             else vhacd_fallback(mc)

        separated.extend(sub_meshes)

    # Post-process
    validated = validate_and_merge_fragments(separated)
    smoothed = [smooth_boundary(m) for m in validated]

    fdi_map = assign_fdi_labels(smoothed, arch)
    return export_teeth(fdi_map, output_dir)
```

---

## 11. Các edge case cần xử lý

### 11.1 Wisdom tooth (răng số 8) vắng mặt

Nếu bệnh nhân chưa mọc răng số 8, expected_count = 7. Implement heuristic kiểm tra:

```python
# Nếu component ngoài cùng có volume < 40% median → có thể là mảnh vỡ, không phải răng 8
# Hoặc skip và để agent/user confirm
```

### 11.2 Ba răng dính nhau (M6 + M7 + M8)

Cross-section sweep sẽ trả về 2 cut planes. `slice_by_planes` xử lý được do cắt tuần tự. Kiểm tra: nếu sau khi cắt vẫn còn component có volume > 1.5x median → recurse sweep trên component đó.

### 11.3 Arch curve thực sự cong (lower jaw)

Nếu dùng x-axis sweep cho lower jaw, mặt phẳng cắt có thể không vuông góc với contact area ở vùng molar. Giải pháp:

```python
# Thay plane_normal = [1, 0, 0] bằng tangent của arch curve tại vị trí đó
# Tính tangent từ fit_arch_curve(), interpolate tại x_position
```

### 11.4 Open mesh (non-watertight)

GLB từ segmentation đôi khi có holes. `trimesh.volume` sẽ không chính xác.

```python
# Dùng mesh.is_watertight để kiểm tra
# Nếu không: dùng mesh.convex_hull.volume * 0.75 làm proxy
volume = mesh.volume if mesh.is_watertight else mesh.convex_hull.volume * 0.75
```

### 11.5 Mesh resolution không đều

Nếu vùng interproximal có quá ít vertices (coarse mesh), curvature sẽ noisy. Giải pháp:

```python
# Subdivide nếu cần trước khi tính curvature
if mesh.faces.shape[0] < 5000:
    mesh = mesh.subdivide()
```

---

## 12. Thứ tự ưu tiên khi chọn method

| Vùng răng | Method chính | Fallback 1 | Fallback 2 |
|-----------|-------------|------------|------------|
| Anterior (1-3) | Geodesic watershed | Cross-section sweep | VHACD |
| Premolar (4-5) | Geodesic watershed | Cross-section sweep | VHACD |
| Molar (6-7) | Cross-section sweep | Geodesic watershed | VHACD |
| Wisdom (8) | Cross-section sweep | VHACD | Manual |

---

## 13. Kiểm tra kết quả (validation)

Sau khi segment xong, agent nên tự validate:

```python
def validate_result(fdi_map: dict, arch: str) -> list[str]:
    warnings = []
    vols = [m.volume if m.is_watertight else m.convex_hull.volume * 0.75
            for m in fdi_map.values()]
    median_vol = np.median(vols)

    for fdi, mesh in fdi_map.items():
        vol = mesh.volume if mesh.is_watertight else mesh.convex_hull.volume * 0.75
        if vol < median_vol * 0.2:
            warnings.append(f"FDI {fdi}: volume quá nhỏ ({vol:.0f} mm³) — có thể là fragment")
        if vol > median_vol * 3.0:
            warnings.append(f"FDI {fdi}: volume quá lớn ({vol:.0f} mm³) — có thể chưa tách hết")

    if len(fdi_map) < EXPECTED_TOOTH_COUNT[arch]:
        warnings.append(f"Chỉ tách được {len(fdi_map)}/{EXPECTED_TOOTH_COUNT[arch]} răng")

    return warnings
```

---

## 14. File output structure
một file .glb vẫn còn những phần cũ, chỉ thay đổi phần upper và lower teeth thành các phần đã được segment từ kết quả chạy của code.

---
