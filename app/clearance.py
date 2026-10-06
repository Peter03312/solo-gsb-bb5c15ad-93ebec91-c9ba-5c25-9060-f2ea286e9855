"""轮廓边界间隙查询（只读演算）。

修复师选择**一个已固化的装配版本快照**与若干陶片对，系统依据该版本快照
中每片陶片的固定轮廓与位姿 (x, y, theta) 计算：

- 两片轮廓**边界之间的最短间隙**（毫米，世界坐标系）；
- 间隙最短处分别落在两片轮廓边界上的**最近点坐标**；
- 两片轮廓是否发生**内部重叠**（面积重叠，允许共边/共点）。

间隙语义（务必区分）：

- ``gap_mm > 0``：两片分离，该值即可用间隙；
- ``gap_mm == 0``：两片边界接触（共边或共点），没有可用间隙；
- ``gap_mm == -1``：仅用于标识两片发生内部重叠，**负值不是深度量度、
  不得当作可用间隙**，实际重叠程度另见 ``overlap_area_mm2``。

结果按 ``gap_mm`` 从小到大（重叠在前、接触其次、分离在后），
间隙相同按陶片编号（小 id 在前）排序。

本模块只读取版本快照，不改变任何位姿、关系或历史；请求存在缺陶片、
重复陶片对、跨版本或无效编号时整次拒绝并逐对指出字段。
"""
from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import nearest_points

from . import geometry

# 内部重叠时间隙字段的标识值：负值仅表示「重叠」，不是可用间隙/穿透深度
OVERLAP_GAP_MM = -1.0
# 边界距离 ≤ 该容差视为接触（间隙为零，mm）
CONTACT_GAP_TOL = 1e-6
_COORD_DIGITS = 4


class ClearanceError(Exception):
    """间隙查询请求级错误；status 为对应 HTTP 状态码。"""

    def __init__(self, status: int, code: str, message: str, *,
                 field: str | None = None, errors: list[dict] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field
        self.errors = errors

    def detail(self) -> dict:
        d = {"message": self.message, "code": self.code}
        if self.field is not None:
            d["field"] = self.field
        if self.errors:
            d["errors"] = self.errors
        return d


def _field(index: int, name: str | None = None) -> str:
    """请求字段路径，如 pairs[2].fragment_a。"""
    return f"pairs[{index}]" + (f".{name}" if name else "")


def _err(index: int, name: str | None, code: str, message: str) -> dict:
    return {"index": index, "field": _field(index, name),
            "code": code, "message": message}


def _placed_polygon(fragment: dict) -> Polygon:
    """按版本快照中的固定位姿把局部闭合轮廓放到世界坐标系。"""
    pose = (fragment["x"], fragment["y"], fragment["theta"])
    return Polygon(geometry.contour_world(fragment["contour"], pose))


def _r4(v: float) -> float:
    v = round(v, _COORD_DIGITS)
    return 0.0 if v == 0 else v   # 归一 -0.0


def _measure_one(index: int, fa: dict, fb: dict) -> dict:
    """计算一对陶片的边界最短间隙、最近点与内部重叠状态（世界坐标，mm）。"""
    id_a, id_b = sorted((fa["id"], fb["id"]))
    if id_a == fa["id"]:
        name_a, name_b = fa["name"], fb["name"]
    else:
        name_a, name_b = fb["name"], fa["name"]
    pa, pb = _placed_polygon(fa), _placed_polygon(fb)
    ba, bb = pa.boundary, pb.boundary

    # 最近点恒在两条边界之间求取：分离时即间隙两端点；接触时为共边/共点；
    # 内部重叠（含一片完全包含另一片）时为两边界彼此最接近的位置。
    qa, qb = nearest_points(ba, bb)
    # qa/qb 的归属与输入 fa/fb 一致；输出按编号小前大后规整。
    if id_a == fa["id"]:
        pt_a, pt_b = qa.coords[0], qb.coords[0]
    else:
        pt_a, pt_b = qb.coords[0], qa.coords[0]
    point_a = [_r4(pt_a[0]), _r4(pt_a[1])]
    point_b = [_r4(pt_b[0]), _r4(pt_b[1])]

    overlap_area = pa.intersection(pb).area
    if overlap_area > geometry.OVERLAP_AREA_TOL:
        return {
            "index": index,
            "fragment_a": id_a, "fragment_b": id_b,
            "name_a": name_a, "name_b": name_b,
            "state": "overlap",
            "gap_mm": OVERLAP_GAP_MM,
            "contact": False, "overlap": True,
            "usable_clearance": False,
            "nearest_point_a": point_a, "nearest_point_b": point_b,
            "overlap_area_mm2": _r4(overlap_area),
        }

    dist = ba.distance(bb)
    if dist <= CONTACT_GAP_TOL:
        return {
            "index": index, "state": "contact",
            "fragment_a": id_a, "fragment_b": id_b,
            "name_a": name_a, "name_b": name_b,
            "gap_mm": 0.0,
            "contact": True, "overlap": False,
            "usable_clearance": False,
            "nearest_point_a": point_a, "nearest_point_b": point_b,
            "overlap_area_mm2": 0.0,
        }
    return {
        "index": index, "state": "separated",
        "fragment_a": id_a, "fragment_b": id_b,
        "name_a": name_a, "name_b": name_b,
        "gap_mm": _r4(dist),
        "contact": False, "overlap": False,
        "usable_clearance": True,
        "nearest_point_a": point_a, "nearest_point_b": point_b,
        "overlap_area_mm2": 0.0,
    }


def measure_clearance(d, version: int, pairs: list[tuple]) -> dict:
    """针对单个版本快照批量计算陶片对边界间隙（只读，不写任何数据）。

    pairs: [(fragment_a, fragment_b, pair_version | None), ...]

    - 目标版本快照不存在 → ClearanceError(404, version_not_found)；
    - 陶片对为空、无效编号、同片成对、快照中缺少陶片、跨版本版本号不一致、
      重复陶片对（顺序相反也算）→ ClearanceError(400, rejected)，
      detail.errors[] 逐对给出 index / field / code / message，整次拒绝。
    """
    snap = d.get_snapshot_state(version)
    if snap is None:
        raise ClearanceError(
            404, "version_not_found",
            f"目标版本 v{version} 的快照不存在：仅允许查询已固化的同一版本快照",
            field="version")

    if not pairs:
        raise ClearanceError(
            400, "empty_pairs", "陶片对列表为空：至少需要提供一对陶片",
            field="pairs")

    fragments = {int(f["id"]): f for f in snap["fragments"]}
    errors: list[dict] = []
    seen: dict[tuple[int, int], int] = {}
    valid: list[tuple[int, int, int]] = []

    for index, (raw_a, raw_b, pair_version) in enumerate(pairs):
        a, b = raw_a, raw_b
        # 跨版本：某对自带的版本号与顶层查询版本不一致（仅允许查询同一快照）
        if pair_version is not None and pair_version != version:
            errors.append(_err(
                index, "version", "cross_version",
                f"第 {index + 1} 对指定版本 v{pair_version}，"
                f"与查询版本 v{version} 不一致：一次查询仅允许使用同一版本快照"))
        # 无效编号：编号必须为正整数
        if a <= 0 or b <= 0:
            bad = "fragment_a" if a <= 0 else "fragment_b"
            bad_id = a if a <= 0 else b
            errors.append(_err(
                index, bad, "invalid_fragment_id",
                f"第 {index + 1} 对陶片编号 {bad_id} 无效：编号必须为正整数"))
            continue
        # 同片成对
        if a == b:
            errors.append(_err(
                index, None, "invalid_pair",
                f"第 {index + 1} 对的两片陶片相同（均为 #{a}）：一对必须是两片不同陶片"))
            continue
        # 重复陶片对（(1,2) 与 (2,1) 视为同一对）
        key = (min(a, b), max(a, b))
        if key in seen:
            errors.append(_err(
                index, None, "duplicate_pair",
                f"第 {index + 1} 对 (#{key[0]}, #{key[1]}) 与第 "
                f"{seen[key] + 1} 对重复：同一陶片对在一次查询中只能出现一次"))
            continue
        seen[key] = index
        # 缺少陶片：编号在该版本快照中不存在
        for side, fid in (("fragment_a", a), ("fragment_b", b)):
            if fid not in fragments:
                errors.append(_err(
                    index, side, "fragment_missing",
                    f"第 {index + 1} 对引用的陶片 #{fid} "
                    f"在版本 v{version} 快照中不存在"))
        if a in fragments and b in fragments:
            valid.append((index, a, b))

    if errors:
        raise ClearanceError(
            400, "rejected",
            f"间隙查询被整次拒绝：{len(errors)} 处字段问题（未执行任何计算，"
            "位姿、关系与历史均不变）",
            errors=errors)

    results = [_measure_one(index, fragments[a], fragments[b])
               for index, a, b in valid]
    # 间隙从小到大；间隙相同按陶片编号（小 id、再大 id）排序
    results.sort(key=lambda r: (r["gap_mm"], r["fragment_a"], r["fragment_b"]))
    return {
        "ok": True,
        "version": version,
        "pair_count": len(results),
        "results": results,
    }
