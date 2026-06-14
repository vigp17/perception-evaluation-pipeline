#!/usr/bin/env python
"""
Core multi-object-tracking evaluation logic, isolated for unit testing without
a GPU or nuScenes.

Per-scene accumulation: each scene gets its own MOTAccumulator so frame
timelines and track ids never bleed across scene boundaries. Scenes are merged
with motmetrics' own multi-accumulator compute at the end (the standard way to
aggregate MOTChallenge sequences).
"""
import numpy as np
import motmetrics as mm


def iou_distance_matrix(gt_boxes, pred_boxes, iou_thr=0.5):
    """Cost matrix motmetrics expects: 1-IoU, NaN where IoU<thr."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))
    g = np.array(gt_boxes, dtype=float)
    p = np.array(pred_boxes, dtype=float)
    ious = np.zeros((len(g), len(p)))
    for i in range(len(g)):
        for j in range(len(p)):
            ix1, iy1 = max(g[i,0], p[j,0]), max(g[i,1], p[j,1])
            ix2, iy2 = min(g[i,2], p[j,2]), min(g[i,3], p[j,3])
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw*ih
            ua = ((g[i,2]-g[i,0])*(g[i,3]-g[i,1])
                  + (p[j,2]-p[j,0])*(p[j,3]-p[j,1]) - inter)
            ious[i,j] = inter/ua if ua > 0 else 0.0
    dist = 1.0 - ious
    dist[ious < iou_thr] = np.nan
    return dist


class TrackingEvaluator:
    """Accumulate frames grouped by scene; summarize merged MOT metrics."""

    def __init__(self, iou_thr=0.5):
        self.iou_thr = iou_thr
        self.accs = []          # one MOTAccumulator per scene
        self.names = []
        self._cur = None
        self._frame = 0

    def new_scene(self, name):
        self._cur = mm.MOTAccumulator(auto_id=False)
        self.accs.append(self._cur)
        self.names.append(name)
        self._frame = 0

    def add_frame(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        if self._cur is None:
            self.new_scene(f"scene_{len(self.accs)}")
        dist = iou_distance_matrix(gt_boxes, pred_boxes, self.iou_thr)
        self._cur.update(gt_ids, pred_ids, dist, frameid=self._frame)
        self._frame += 1

    def summary(self):
        mh = mm.metrics.create()
        if not self.accs:
            return None
        report = mh.compute_many(
            self.accs, names=self.names,
            metrics=['mota', 'motp', 'num_switches', 'num_fragmentations',
                     'mostly_tracked', 'mostly_lost', 'num_false_positives',
                     'num_misses', 'num_objects', 'idf1'],
            generate_overall=True)
        return report.loc['OVERALL']


def _test():
    # perfect tracking across 2 scenes -> MOTA 1.0, 0 switches
    ev = TrackingEvaluator()
    for s in range(2):
        ev.new_scene(f"s{s}")
        for f in range(5):
            ev.add_frame([1, 2],
                         [[10+f*5,10,30+f*5,30],[100,100,120,120]],
                         [101,102],
                         [[10+f*5,10,30+f*5,30],[100,100,120,120]])
    r = ev.summary()
    print(f"Perfect 2-scene: MOTA={r['mota']:.3f} switches={int(r['num_switches'])}")
    assert r['mota'] == 1.0 and r['num_switches'] == 0

    # reused GT id across scenes must NOT create a phantom switch
    ev2 = TrackingEvaluator()
    for s in range(2):
        ev2.new_scene(f"s{s}")
        for f in range(5):
            ev2.add_frame([1], [[10+f*5,10,30+f*5,30]], [5],
                          [[10+f*5,10,30+f*5,30]])
    r2 = ev2.summary()
    print(f"Reused ids across scenes: switches={int(r2['num_switches'])} (want 0)")
    assert r2['num_switches'] == 0

    # injected switch within a scene still caught
    ev3 = TrackingEvaluator(); ev3.new_scene("s")
    for f in range(5):
        pid = 101 if f < 3 else 999
        ev3.add_frame([1],[[10+f*5,10,30+f*5,30]],[pid],[[10+f*5,10,30+f*5,30]])
    r3 = ev3.summary()
    print(f"Injected switch: switches={int(r3['num_switches'])} (want 1)")
    assert r3['num_switches'] == 1
    print("UNIT TESTS PASSED")


if __name__ == '__main__':
    _test()
