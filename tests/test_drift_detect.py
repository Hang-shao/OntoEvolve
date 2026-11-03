"""模块 D（三信号漂移检测）的单元测试：淘汰 / 重定义 / 涌现。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.controlled_stream import Occurrence
from src.drift_detect import DriftDetector
from src.ontology_model import Axiom, Concept, Ontology


def _unit(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    v[i % dim] = 1.0
    return v


def _onto() -> Ontology:
    c0 = Concept(cid="C0", name="Dog", parent="Animal", centroid=_unit(0),
                 axioms=[Axiom(kind="OP", subj="C0", pred="livesIn", obj="Forest",
                               confidence=1.0),
                         Axiom(kind="DP", subj="C0", pred="size", obj="large",
                               confidence=1.0)])
    c1 = Concept(cid="C1", name="Car", parent="Vehicle", centroid=_unit(1),
                 axioms=[Axiom(kind="OP", subj="C1", pred="madeOf", obj="Steel",
                               confidence=1.0)])
    return Ontology({"C0": c0, "C1": c1})


def _occ(leaf: str, vec: np.ndarray, op_slots=None, dp_slots=None) -> Occurrence:
    return Occurrence(iid=0, leaf=leaf, variant="base", text="x", vec=vec,
                      op_slots=op_slots or [], dp_slots=dp_slots or [], sup="S")


def test_extinction_detected_after_window():
    params = C.PARAMS.copy(window_w=3, eps_ext=0.5, base_rate=2.0,
                           theta_red=99.0, theta_sem=99.0)
    det = DriftDetector(_onto(), params)
    alarms = []
    for t in range(1, 7):
        items = [_occ("Dog", _unit(0)) for _ in range(3)] if t <= 3 else []
        alarms.extend(det.observe(t, items))
    ext = [a for a in alarms if a.dtype == "extinction" and a.cid == "C0"]
    assert ext, "淘汰未被检出"
    assert ext[0].t >= 5          # 窗口 W=3 之后才可能判定


def test_redefinition_detected_via_axiom_signature():
    params = C.PARAMS.copy(theta_red=0.5, tau_conf=0.4, window_w=99,
                           theta_sem=99.0, base_rate=3.0)
    det = DriftDetector(_onto(), params)
    alarms = []
    for t in range(1, 6):
        slots = [("livesIn", "Forest")] if t <= 3 else [("eats", "Meat")]
        items = [_occ("Dog", _unit(0), op_slots=slots) for _ in range(3)]
        alarms.extend(det.observe(t, items))
    red = [a for a in alarms if a.dtype == "redefinition"]
    assert red, "重定义未被检出"
    assert red[0].cid == "C0"


def test_emergence_detected_for_unlinked_persistent_cluster():
    params = C.PARAMS.copy(kappa=2, min_new_size=3, theta_sem=0.1,
                           window_w=99, theta_red=99.0)
    det = DriftDetector(_onto(), params)
    alarms = []
    for t in range(1, 5):
        items = [_occ("Rocket", _unit(5)) for _ in range(2)]
        alarms.extend(det.observe(t, items))
    eme = [a for a in alarms if a.dtype == "emergence"]
    assert eme, "涌现未被检出"
    assert eme[0].cid.startswith("NEW")
    assert eme[0].t >= params.kappa


def test_no_alarm_when_stream_is_stable():
    params = C.PARAMS.copy(window_w=3, eps_ext=0.5, theta_red=99.0,
                           theta_sem=99.0, theta_shift=99.0)
    det = DriftDetector(_onto(), params)
    alarms = []
    for t in range(1, 8):
        items = [_occ("Dog", _unit(0), op_slots=[("livesIn", "Forest")])
                 for _ in range(3)]
        alarms.extend(det.observe(t, items))
    # C1 从未收到实例，按定义确实“被淘汰”；这里只检查持续有实例支撑的 C0
    assert not [a for a in alarms if a.cid == "C0"], f"C0 产生误报: {alarms}"
