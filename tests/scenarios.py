"""Scenario library for the room simulation (shared by tests and the exploration script)."""
from __future__ import annotations

import numpy as np

import synth
from synth import HAND, RELEASE, RIM_C, Sim, hold, throw


def shot(offset=(0, 0, 0), flight=0.8, seed=0, hold_s=0.8, extra=1.6) -> Sim:
    return synth.shot(offset, flight, hold_s=hold_s, seed=seed, extra=extra)


def find_shot(pred, offsets, flights, seed=0) -> Sim:
    """First (offset, flight) whose simulated outcome satisfies pred."""
    for fl in flights:
        for off in offsets:
            s = shot(off, fl, seed=seed)
            if pred(s):
                return s
    raise RuntimeError("no trajectory found")


def rim_in() -> Sim:
    offs = [(dx, 0, dz) for dz in np.linspace(-0.07, 0.07, 15) for dx in (0.0, 0.03, -0.03)]
    return find_shot(lambda s: s.made and s.rim_hits >= 1, offs, [0.8, 0.7, 0.9])


def front_rim_miss() -> Sim:
    offs = [(0, 0, dz) for dz in np.linspace(-0.08, -0.2, 13)]
    return find_shot(lambda s: not s.made and s.rim_hits >= 1, offs, [0.8, 0.7])


def side_miss(sign=1) -> Sim:
    return find_shot(lambda s: not s.made and s.rim_hits == 0, [(sign * dx, 0, 0) for dx in (0.3, 0.35, 0.4)], [0.8])


def airball_short() -> Sim:
    return find_shot(lambda s: not s.made and s.rim_hits == 0, [(0, -0.35, -0.6), (0, -0.3, -0.5)], [0.7, 0.8])


def long_off_wall() -> Sim:
    return find_shot(lambda s: not s.made and s.wall_hits >= 1, [(0, 0.35, 0.3), (0, 0.4, 0.35), (0.1, 0.45, 0.3)], [0.8, 0.7])


# ---------------------------------------------------------------- non-shots

def holding(seed=0) -> Sim:
    s = Sim()
    hold(s, HAND, HAND + np.array([0.03, 0.02, 0.0]), 1.2, t0=0.0, seed=seed, jitter=0.01)
    hold(s, HAND + np.array([0.03, 0.02, 0.0]), HAND, 1.2, seed=seed, jitter=0.01)
    return s


def moving_in_front(seed=0) -> Sim:
    s = Sim()
    p = [HAND, HAND + np.array([-0.3, 0.1, 0.05]), HAND + np.array([0.15, 0.25, -0.05]), HAND + np.array([0.25, -0.05, 0.1]), HAND]
    t = 0.0
    for a, b in zip(p, p[1:]):
        t = hold(s, a, b, 0.6, t0=None if s.t else 0.0, seed=seed) + 0.001
    return s


def dribble(seed=0, n=5) -> Sim:
    s = Sim()
    top = np.array([0.15, -0.35, 1.3])
    floor = np.array([0.15, synth.FLOOR_Y + synth.R_BALL, 1.35])
    hold(s, top, top, 0.5, t0=0.0, seed=seed)
    for _ in range(n):
        hold(s, top, floor, 0.22, seed=seed, jitter=0)
        hold(s, floor, top, 0.28, seed=seed, jitter=0)
    return s


def catch(seed=0) -> Sim:
    """Ball comes back toward the camera (e.g. passed back from the wall) and is caught."""
    s = Sim()
    far = np.array([0.0, 0.1, 1.45])
    hold(s, far, far, 0.3, t0=0.0, seed=seed)
    hold(s, far, HAND, 0.45, seed=seed, jitter=0)
    hold(s, HAND, HAND, 0.8, seed=seed)
    return s


def pump_fake(seed=0) -> Sim:
    s = Sim()
    up = RELEASE + np.array([0.0, 0.08, -0.05])
    hold(s, HAND, HAND, 0.8, t0=0.0, seed=seed)
    hold(s, HAND, up, 0.16, seed=seed, jitter=0)
    hold(s, up, HAND, 0.25, seed=seed, jitter=0)
    hold(s, HAND, HAND, 0.8, seed=seed)
    return s


def retrieval(seed=0) -> Sim:
    """Ball rolls back from under the hoop and is picked up and brought to the chest."""
    s = Sim()
    a = np.array([0.1, synth.FLOOR_Y + synth.R_BALL, 1.5])
    b = np.array([0.15, synth.FLOOR_Y + synth.R_BALL, 0.9])
    hold(s, a, b, 1.0, t0=0.0, seed=seed, jitter=0)
    hold(s, b, HAND, 0.6, seed=seed, jitter=0)
    hold(s, HAND, HAND, 0.6, seed=seed)
    return s


def consecutive(sims: list[Sim], gap: float = 0.5) -> Sim:
    """Concatenate scenarios in time (same ball)."""
    out = Sim()
    t0 = 0.0
    for s in sims:
        for t, P, h in zip(s.t, s.P, s.held):
            out.t.append(t0 + t)
            out.P.append(P)
            out.held.append(h)
        if s.release_t is not None and out.release_t is None:
            out.release_t = t0 + s.release_t
        t0 = out.t[-1] + gap
        # walk back to the shooting position during the gap
        last = out.P[-1]
        hold(out, last, HAND, gap * 0.9, t0=out.t[-1] + 0.001)
    return out
