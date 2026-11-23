import math

import numpy as np
import pytest

from calibration import CameraMotionMonitor, HoopCalibration, estimate_shift, gray_thumb
from geometry import Ellipse, ellipse_from_rim_points, point_in_polygon, segment_crosses_opening, segment_intersects_ellipse


def test_ellipse_from_four_rim_clicks():
    e = ellipse_from_rim_points((100, 200), (300, 200), (200, 170), (200, 230))
    assert e.cx == pytest.approx(200) and e.cy == pytest.approx(200)
    assert e.a == pytest.approx(100) and e.b == pytest.approx(30)
    assert e.angle == pytest.approx(0)
    tilted = ellipse_from_rim_points((100, 200), (300, 240), (195, 190), (205, 250))
    assert math.degrees(tilted.angle) == pytest.approx(11.31, abs=0.1)


def test_local_coordinates_round_trip_and_contains():
    e = Ellipse(200, 150, 80, 20, 0.3)
    u, v = e.to_local(230, 170)
    x, y = e.from_local(u, v)
    assert (x, y) == pytest.approx((230, 170))
    assert e.contains(200, 150)
    assert not e.contains(200, 150 + 60)


def test_fast_ball_crossing_between_two_frames():
    e = Ellipse(200, 150, 80, 20, 0.0)
    # above the rim in one frame, below it in the next: still a crossing of the opening
    ok, off = segment_crosses_opening(e, (205, 80), (195, 230))
    assert ok and off < 0.1
    # same vertical travel but beside the rim
    ok, off = segment_crosses_opening(e, (300, 80), (300, 230))
    assert not ok and off > 1
    # moving UP through the opening is never a crossing
    ok, _ = segment_crosses_opening(e, (195, 230), (205, 80))
    assert not ok
    assert segment_intersects_ellipse(e, (100, 150), (300, 150))


def test_point_in_polygon():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon(5, 5, sq) and not point_in_polygon(15, 5, sq)


def test_calibration_zones_save_load_and_scale(tmp_path):
    cal = HoopCalibration.from_points(1280, 720, (600, 200), (700, 200), (650, 185), (650, 215))
    e = cal.ellipse
    above = e.from_local(0, -e.b - 0.5 * e.a)
    below = e.from_local(0, e.b + 0.5 * e.a)
    assert point_in_polygon(*above, cal.zone_a) and not point_in_polygon(*above, cal.zone_c)
    assert point_in_polygon(*below, cal.zone_c) and not point_in_polygon(*below, cal.zone_a)
    cal.ignore_regions.append([[0, 0], [100, 0], [100, 100], [0, 100]])
    p = tmp_path / "hoop.json"
    cal.save(str(p))
    back = HoopCalibration.load(str(p))
    assert back.ellipse.cx == pytest.approx(650) and len(back.ignore_regions) == 1
    small = back.scaled_to(640, 360)
    assert small.ellipse.cx == pytest.approx(325) and small.ellipse.a == pytest.approx(25)
    assert back.aspect_matches(640, 360) and not back.aspect_matches(640, 480)


def _scene(dx=0, dy=0, blob=None):
    yy, xx = np.mgrid[0:360, 0:640].astype(np.float32)
    img = 128 + 60 * np.sin((xx - dx) * 0.05) * np.cos((yy - dy) * 0.04) + 40 * np.sin((xx + yy - dx - dy) * 0.02)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if blob is not None:
        import cv2

        cv2.circle(img, blob, 40, 10, -1)
    return np.dstack([img] * 3)


def test_camera_movement_is_detected_but_people_walking_by_are_not():
    ref = gray_thumb(_scene())
    dx, dy, _, _ = estimate_shift(ref, gray_thumb(_scene(20, 0)))
    assert (dx, dy) != (0, 0)
    m = CameraMotionMonitor(ref)
    for x in range(100, 600, 60):
        assert not m.check(_scene(blob=(x, 200)))
    moved = [m.check(_scene(25, 10)) for _ in range(4)]
    assert moved[-1]
