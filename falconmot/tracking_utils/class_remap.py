"""
class_remap.py — Pluggable class-merging module for multi-class MOT evaluation.

Vấn đề
------
track.py hiện dùng model train trên 10 class VisDrone nhưng muốn evaluate trên
taxonomy rút gọn (5 class). Cách cũ (`_EVAL_SKIP_1IDX` trong io.py /
`_EVAL_SKIP_0IDX` trong track.py) chỉ SKIP (drop hẳn) các class dư — ở CẢ HAI
phía GT và prediction trước khi đưa vào motmetrics accumulator.

Hệ quả nhiễu: nếu model detect đúng vị trí nhưng gán nhãn-con khác trong cùng
một nhóm ngữ nghĩa (ví dụ GT="van", model dự đoán "truck" tại đúng vị trí đó),
SKIP làm GT "van" bị bỏ qua hoàn toàn → không có GT để match → prediction
"truck" hoá thành unmatched hypothesis (FP giả), đồng thời GT "van" cũng không
tính FN (vì đã bị lọc trước) nhưng cũng không được tính đúng (TP) dù model làm
đúng việc. Ngược lại nếu GT="van" bị skip mà KHÔNG merge, một detection đúng bị
vứt bỏ không tính. → MOTA/IDF1 bị nhiễu bởi nhầm lẫn nhãn-con trong cùng nhóm.

Giải pháp: MERGE thay vì SKIP cho các class có nhóm tương đương ngữ nghĩa rõ
ràng (van+truck, tricycle+awning-tricycle, pedestrian+people) — gộp cả GT và
prediction về cùng 1 target id trước khi match, để những trường hợp "đúng vị
trí, nhầm nhãn-con trong nhóm" KHÔNG còn bị tính nhiễu. Những class không có
nhóm tương đương rõ trong taxonomy target (bicycle, motor) vẫn DROP — vì ép
gộp khập khiễng (ví dụ nhét vào "car") sẽ tạo FN giả mới do model chưa từng
học match bicycle/motor ↔ car.

Thiết kế
--------
Module này KHÔNG đụng tới model/architecture/training — chỉ là một lớp remap
thuần áp dụng tại thời điểm đọc GT (io.py) và ghi prediction (track.py), nên
có thể gắn rời / tháo rời (`set_merge_profile(None)` = passthrough, giữ đúng
hành vi cũ cho '10class' / '5class' / '4class' skip-mode).

Sử dụng
-------
    from falconmot.tracking_utils import class_remap

    class_remap.set_merge_profile('5class_merge')   # bật merge
    class_remap.set_merge_profile(None)              # tắt merge (passthrough)

    target = class_remap.remap_raw_cls_id(raw_id_0idx)
    # -> target 0-indexed id (nếu giữ), hoặc raw_id_0idx nguyên bản (nếu tắt
    #    merge), hoặc None (nếu profile đang active drop class này)
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# VisDrone gốc — 10 class, 0-indexed (khớp cls2id trong scripts/gen_dataset_visdrone.py
# và VISDRONE_DET_CATEGORIES trong scripts/visdrone_det_to_coco.py)
# ---------------------------------------------------------------------------
VISDRONE_ID2NAME = {
    0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
    5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor',
}
VISDRONE_NAME2ID = {v: k for k, v in VISDRONE_ID2NAME.items()}
NUM_SOURCE_CLASSES = len(VISDRONE_ID2NAME)


class ClassMergeProfile:
    """Một cách gộp N class gốc -> M class target (M <= N).

    raw_to_target_name : dict[raw_id_0idx] -> target_name | None
                          None = drop hẳn (không có nhóm tương đương ở target)
    target_order        : thứ tự target class -> định nghĩa target id 0..M-1
    """

    def __init__(self, name: str, raw_to_target_name: Dict[int, Optional[str]],
                 target_order: List[str]):
        self.name = name
        self.target_order = list(target_order)
        self.target_name2id = {n: i for i, n in enumerate(self.target_order)}
        self.raw_to_target_name = dict(raw_to_target_name)

        missing = set(range(NUM_SOURCE_CLASSES)) - set(self.raw_to_target_name)
        if missing:
            missing_names = [VISDRONE_ID2NAME[i] for i in sorted(missing)]
            raise ValueError(f"[{name}] thieu mapping cho raw class: {missing_names}")
        for raw_id, tgt_name in self.raw_to_target_name.items():
            if tgt_name is not None and tgt_name not in self.target_name2id:
                raise ValueError(
                    f"[{name}] target '{tgt_name}' (tu raw={VISDRONE_ID2NAME[raw_id]}) "
                    f"khong nam trong target_order={self.target_order}")

        self.raw_to_target_id: Dict[int, Optional[int]] = {
            raw_id: (self.target_name2id[tgt_name] if tgt_name is not None else None)
            for raw_id, tgt_name in self.raw_to_target_name.items()
        }

    @property
    def num_target_classes(self) -> int:
        return len(self.target_order)

    def remap(self, raw_cls_id_0idx: int) -> Optional[int]:
        """raw 0-indexed -> target 0-indexed, hoặc None nếu profile drop class này."""
        return self.raw_to_target_id.get(raw_cls_id_0idx)

    def describe(self) -> str:
        groups: Dict[str, List[str]] = {n: [] for n in self.target_order}
        dropped = []
        for raw_id, tgt in self.raw_to_target_name.items():
            raw_name = VISDRONE_ID2NAME[raw_id]
            (dropped if tgt is None else groups[tgt]).append(raw_name)
        lines = [f"[{self.name}] {self.num_target_classes} target class "
                 f"(tu {NUM_SOURCE_CLASSES} class goc):"]
        for tgt in self.target_order:
            lines.append(f"  {tgt:<12s} <- {' + '.join(groups[tgt])}")
        if dropped:
            lines.append(f"  (dropped, khong co nhom tuong duong: {', '.join(dropped)})")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Profile: 5class_merge_benchmark
#   pedestrian <- pedestrian + people
#   car        <- car
#   truck      <- truck + van
#   tricycle   <- tricycle + awning-tricycle
#   bus        <- bus
#   (drop: bicycle, motor — không có nhóm tương đương trong 5 target trên,
#    ép gộp sẽ tạo FN giả vì model chưa từng học match bicycle/motor ↔ car/van)
# ---------------------------------------------------------------------------
VISDRONE_5CLASS_MERGE_BENCHMARK = ClassMergeProfile(
    name='5class_merge_benchmark',
    target_order=['pedestrian', 'car', 'truck', 'tricycle', 'bus'],
    raw_to_target_name={
        VISDRONE_NAME2ID['pedestrian']:      'pedestrian',
        VISDRONE_NAME2ID['people']:          'pedestrian',
        VISDRONE_NAME2ID['bicycle']:         None,
        VISDRONE_NAME2ID['car']:             'car',
        VISDRONE_NAME2ID['van']:             'truck',
        VISDRONE_NAME2ID['truck']:           'truck',
        VISDRONE_NAME2ID['tricycle']:        'tricycle',
        VISDRONE_NAME2ID['awning-tricycle']: 'tricycle',
        VISDRONE_NAME2ID['bus']:             'bus',
        VISDRONE_NAME2ID['motor']:           None,
    },
)

_PROFILES: Dict[str, ClassMergeProfile] = {
    '5class_merge_benchmark': VISDRONE_5CLASS_MERGE_BENCHMARK,
}

# ---------------------------------------------------------------------------
# Profile: 5class_merge_competition — mirrors '4class' competition taxonomy
# nhưng tách "car" cũ (vốn là catch-all car+van+truck+...) thành 2 nhóm mịn
# hơn: car (chỉ sedan) và truck (van+truck):
#   person     <- pedestrian + people
#   car        <- car
#   truck      <- van + truck
#   motorcycle <- motor
#   bicycle    <- bicycle
#   (drop: tricycle, awning-tricycle, bus — không có nhóm tương đương trong
#    5 target trên; KHÔNG gộp vào truck để tránh kéo lệch benchmark — xem
#    yêu cầu loại 3 class này khỏi tính toán hoàn toàn)
# ---------------------------------------------------------------------------
VISDRONE_5CLASS_MERGE_COMPETITION = ClassMergeProfile(
    name='5class_merge_competition',
    target_order=['person', 'car', 'truck', 'motorcycle', 'bicycle'],
    raw_to_target_name={
        VISDRONE_NAME2ID['pedestrian']:      'person',
        VISDRONE_NAME2ID['people']:          'person',
        VISDRONE_NAME2ID['bicycle']:         'bicycle',
        VISDRONE_NAME2ID['car']:             'car',
        VISDRONE_NAME2ID['van']:             'truck',
        VISDRONE_NAME2ID['truck']:           'truck',
        VISDRONE_NAME2ID['tricycle']:        None,
        VISDRONE_NAME2ID['awning-tricycle']: None,
        VISDRONE_NAME2ID['bus']:             None,
        VISDRONE_NAME2ID['motor']:           'motorcycle',
    },
)
_PROFILES['5class_merge_competition'] = VISDRONE_5CLASS_MERGE_COMPETITION

_active_profile_name: Optional[str] = None  # None = tắt merge (passthrough)


def register_profile(profile: ClassMergeProfile) -> None:
    """Cho phép thêm profile tùy biến từ bên ngoài module (vd 1 script khác)."""
    _PROFILES[profile.name] = profile


def available_profiles() -> List[str]:
    return list(_PROFILES)


def set_merge_profile(name: Optional[str]) -> None:
    """name=None -> tắt merge, giữ nguyên hành vi class gốc (dùng cho skip-mode cũ)."""
    global _active_profile_name
    if name is not None and name not in _PROFILES:
        raise ValueError(f"Unknown merge profile: {name!r}. Available: {available_profiles()}")
    _active_profile_name = name


def get_active_profile() -> Optional[ClassMergeProfile]:
    return _PROFILES.get(_active_profile_name) if _active_profile_name else None


def remap_raw_cls_id(raw_cls_id_0idx: int) -> Optional[int]:
    """Áp dụng active profile (nếu có) lên 1 raw class id (0-indexed).

    Trả về:
      - target 0-indexed class id, nếu có active profile và class được giữ
      - raw_cls_id_0idx nguyên bản, nếu KHÔNG có active profile (passthrough)
      - None, nếu active profile drop class này
    """
    profile = get_active_profile()
    if profile is None:
        return raw_cls_id_0idx
    return profile.remap(raw_cls_id_0idx)


def num_active_classes(fallback: int) -> int:
    """Số class hiệu lực hiện tại (target nếu có merge, ngược lại fallback)."""
    profile = get_active_profile()
    return profile.num_target_classes if profile else fallback


def target_class_names(fallback: Optional[List[str]] = None) -> List[str]:
    profile = get_active_profile()
    if profile:
        return list(profile.target_order)
    return fallback if fallback is not None else [VISDRONE_ID2NAME[i] for i in range(NUM_SOURCE_CLASSES)]
