#!/usr/bin/env python3
"""
ballpicker_nav.py
Minimal custom Python nav stack for a golf-range ball picker.

Features:
- Boustrophedon (lawnmower) lane generator inside a rectangle
- Pure Pursuit lateral control + PID longitudinal control
- Pluggable I/O: GPS/IMU/Encoders/MCU (stubs included)
- Tiny simulator to test without hardware

No heavy deps: only Python 3 + standard library + (optional) numpy for math speed.
"""

from __future__ import annotations
import math, sys, threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import numpy as np
except ImportError:
    np = None  


# Config


@dataclass
class RobotConfig:
    # (meters)
    wheelbase: float = 1.0         # distance between front & rear axle (or virtual axle)
    max_steer_deg: float = 28.0    # steering limit
    max_speed: float = 1.5         # m/s
    cruise_speed: float = 1.2      # m/s
    lookahead_min: float = 1.0     # m
    lookahead_max: float = 3.0     # m
    lookahead_gain: float = 1.2    # multiply by speed to set dynamic lookahead

    # Controllers
    kp_speed: float = 0.8
    ki_speed: float = 0.2
    kd_speed: float = 0.05

    # Waypoint following
    waypoint_tolerance: float = 0.4  

    # Safety & ops
    estop_enabled: bool = True
    soft_slowdown_radius: float = 2.5  


# Simple Controllers


class PID:
    def __init__(self, kp: float, ki: float, kd: float, i_limit: float = 1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_term = 0.0
        self.prev_err = 0.0
        self.i_limit = i_limit

    def reset(self):
        self.i_term = 0.0
        self.prev_err = 0.0

    def step(self, error: float, dt: float) -> float:
        self.i_term += error * dt
        # clamp integral windup
        self.i_term = max(-self.i_limit, min(self.i_term, self.i_limit))
        d_err = (error - self.prev_err) / dt if dt > 1e-4 else 0.0
        self.prev_err = error
        return self.kp * error + self.ki * self.i_term + self.kd * d_err


class PurePursuit:
    """
    Classic Pure Pursuit lateral controller.
    Given current pose (x, y, yaw) and polyline waypoints, compute steering angle.
    """
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self._last_target_idx = 0

    @staticmethod
    def _dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
        dx, dy = a[0]-b[0], a[1]-b[1]
        return math.hypot(dx, dy)

    def target_lookahead(self, speed: float) -> float:
        Ld = self.cfg.lookahead_gain * max(0.0, speed)
        return max(self.cfg.lookahead_min, min(self.cfg.lookahead_max, Ld))

    def compute(self, pose: Tuple[float,float,float],
                path: List[Tuple[float,float]],
                speed: float) -> Tuple[float, int]:
        """
        Returns (steer_angle_rad, target_index)
        """
        x, y, yaw = pose
        Ld = self.target_lookahead(speed)

        # Find the path point at ~ lookahead distance from current pose
        # Start from last used index to avoid scanning the whole path
        idx_start = self._last_target_idx
        best_idx = idx_start
        best_d = float('inf')

        # If numpy is available, vectorize for speed; else do a simple loop.
        if np is not None:
            pts = np.array(path[idx_start:], dtype=float)
            dists = np.hypot(pts[:,0] - x, pts[:,1] - y)
            # target = first index where distance >= Ld, else the farthest point
            ge = np.where(dists >= Ld)[0]
            if ge.size > 0:
                best_idx = idx_start + int(ge[0])
            else:
                best_idx = len(path) - 1
        else:
            for i in range(idx_start, len(path)):
                d = self._dist((x,y), path[i])
                if abs(d - Ld) < best_d:
                    best_d = abs(d - Ld)
                    best_idx = i

        self._last_target_idx = best_idx

        tx, ty = path[best_idx]
        # Transform target point to vehicle coordinate frame
        dx, dy = tx - x, ty - y
        # Heading to target in map frame
        # Compute lateral error in vehicle frame:
        # x_v = cos(yaw)*dx + sin(yaw)*dy
        # y_v = -sin(yaw)*dx + cos(yaw)*dy
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        x_v =  cos_y * dx + sin_y * dy
        y_v = -sin_y * dx + cos_y * dy

        # Pure pursuit curvature kappa = 2*y_v / (Ld^2)
        # steering = atan(L * kappa)
        if Ld < 1e-4:
            return 0.0, best_idx
        kappa = 2.0 * y_v / (Ld * Ld)
        steer = math.atan(self.cfg.wheelbase * kappa)

        # Clamp to steering limits
        steer = max(-math.radians(self.cfg.max_steer_deg),
                    min( math.radians(self.cfg.max_steer_deg), steer))
        return steer, best_idx


# Coverage Path Generator


def make_lawnmower_rect(xmin: float, xmax: float, ymin: float, ymax: float,
                        lane_spacing: float, lane_direction: str = "+x") -> List[Tuple[float,float]]:
    """
    Simple boustrophedon lanes inside an axis-aligned rectangle.
    lane_direction: "+x" to sweep along +X, "+y" to sweep along +Y.
    """
    assert xmax > xmin and ymax > ymin, "Invalid rectangle"
    assert lane_spacing > 0.1, "lane_spacing too small"

    pts: List[Tuple[float,float]] = []
    if lane_direction == "+x":
        width  = ymax - ymin
        lanes  = max(1, int(width / lane_spacing))
        y = ymin + 0.5 * lane_spacing
        forward = True
        for _ in range(lanes):
            if forward:
                pts.append((xmin, y))
                pts.append((xmax, y))
            else:
                pts.append((xmax, y))
                pts.append((xmin, y))
            forward = not forward
            y += lane_spacing
    else:
        height = xmax - xmin
        lanes  = max(1, int(height / lane_spacing))
        x = xmin + 0.5 * lane_spacing
        forward = True
        for _ in range(lanes):
            if forward:
                pts.append((x, ymin))
                pts.append((x, ymax))
            else:
                pts.append((x, ymax))
                pts.append((x, ymin))
            forward = not forward
            x += lane_spacing
    return pts




@dataclass
class NavState:
    x: float
    y: float
    yaw: float     # radians, 0 = +X
    v: float       # m/s current forward speed

class SensorSuite:
    """
    Replace methods with real drivers:
    - GPS (convert lat/lon to local ENU)
    - IMU yaw
    - Encoders for speed
    """
    def __init__(self):
        self._state = NavState(0.0, 0.0, 0.0, 0.0)
        self._lock = threading.Lock()

    def read(self) -> NavState:
        with self._lock:
            return NavState(self._state.x, self._state.y, self._state.yaw, self._state.v)

    def _sim_update(self, x, y, yaw, v):
        with self._lock:
            self._state = NavState(x, y, yaw, v)


class MCU:
    """
    Replace with serial/CAN to your motor controller.
    For example: send throttle (-1..+1) and steering (radians).
    """
    def __init__(self):
        self.last_cmd = (0.0, 0.0)

    def send(self, throttle_cmd: float, steer_cmd_rad: float):
        throttle_cmd = max(-1.0, min(1.0, throttle_cmd))
        steer_cmd_rad = max(-math.radians(28), min(math.radians(28), steer_cmd_rad))
        self.last_cmd = (throttle_cmd, steer_cmd_rad)

    def estop(self):
        self.last_cmd = (0.0, 0.0)
        # hardware: open safety relay, set brake, etc.


# Navigator


class Navigator:
    def __init__(self, cfg: RobotConfig, sensors: SensorSuite, mcu: MCU, waypoints: List[Tuple[float,float]]):
        self.cfg = cfg
        self.sensors = sensors
        self.mcu = mcu
        self.path = waypoints
        self.latctl = PurePursuit(cfg)
        self.spdctl = PID(cfg.kp_speed, cfg.ki_speed, cfg.kd_speed, i_limit=1.5)
        self.target_idx = 0
        self.done = False

    def step(self, dt: float, soft_obstacle_dist: Optional[float] = None):
        state = self.sensors.read()
        if self.target_idx >= len(self.path):
            self.done = True
            self.mcu.send(0.0, 0.0)
            return

        # Compute steering
        steer_cmd, idx = self.latctl.compute((state.x, state.y, state.yaw), self.path, state.v)
        self.target_idx = idx

        # Compute desired speed
        v_des = self.cfg.cruise_speed
        if soft_obstacle_dist is not None and soft_obstacle_dist < self.cfg.soft_slowdown_radius:
            # linearly reduce speed when close to obstacle
            v_des = max(0.2, self.cfg.cruise_speed * (soft_obstacle_dist / self.cfg.soft_slowdown_radius))
        # Slow near final waypoints
        if idx > len(self.path) - 5:
            v_des = min(v_des, 0.7)
            
        v_err = v_des - state.v
        throttle = self.spdctl.step(v_err, dt)
        throttle = max(-1.0, min(1.0, throttle))

        # Send
        self.mcu.send(throttle, steer_cmd)

        # If we’re very close to the final waypoint, stop
        if idx >= len(self.path) - 1:
            d2final = math.hypot(self.path[-1][0]-state.x, self.path[-1][1]-state.y)
            if d2final < self.cfg.waypoint_tolerance:
                self.done = True
                self.mcu.send(0.0, 0.0)


# Mini Simulator (optional)


class SimpleSim:
    """
    Unicycle / bicycle-ish kinematics to visualize behavior quickly.
    """
    def __init__(self, cfg: RobotConfig, sensors: SensorSuite, mcu: MCU):
        self.cfg = cfg
        self.sensors = sensors
        self.mcu = mcu
        # internal state
        self.x, self.y, self.yaw, self.v = 0.0, 0.0, 0.0, 0.0

    def step(self, dt: float):
        throttle, steer = self.mcu.last_cmd
        # Map throttle (-1..1) to accel; very rough
        accel = 1.5 * throttle - 0.3 * self.v  # drag term
        self.v += accel * dt
        self.v = max(0.0, min(self.cfg.max_speed, self.v))
        # Bicycle model
        beta = 0.0  # no slip
        self.x += self.v * math.cos(self.yaw + beta) * dt
        self.y += self.v * math.sin(self.yaw + beta) * dt
        self.yaw += (self.v / max(1e-3, self.cfg.wheelbase)) * math.tan(steer) * dt
        # Wrap yaw
        self.yaw = (self.yaw + math.pi) % (2*math.pi) - math.pi
        # Push to sensors
        self.sensors._sim_update(self.x, self.y, self.yaw, self.v)


# Main (demo run)


def demo_run():
    import matplotlib.pyplot as plt  # pip install matplotlib if needed

    cfg = RobotConfig()

    # --- coverage geometry ---
    pickup_width = 1.8
    overlap = 0.25
    lane_spacing = pickup_width * (1.0 - overlap)
    xmin, xmax, ymin, ymax = 0.0, 200.0, 0.0, 75.0  # small test area
    path = make_lawnmower_rect(xmin, xmax, ymin, ymax, lane_spacing, lane_direction="+x")

    sensors = SensorSuite()
    mcu = MCU()
    nav = Navigator(cfg, sensors, mcu, path)
    sim = SimpleSim(cfg, sensors, mcu)

    # --- log trajectory for plotting ---
    traj_x, traj_y = [], []

    # Main loop (50 Hz sim, 20 Hz control)
    ctrl_dt = 0.05
    sim_dt  = 0.02
    t = 0.0
    ctrl_acc = 0.0

    print("Running mini-sim… (Ctrl+C to stop)")
    try:
        while not nav.done and t < 20000.0:
            ctrl_acc += sim_dt
            if ctrl_acc >= ctrl_dt:
                nav.step(ctrl_dt, soft_obstacle_dist=None)
                ctrl_acc = 0.0

            sim.step(sim_dt)
            st = sensors.read()
            traj_x.append(st.x); traj_y.append(st.y)
            t += sim_dt

        print("Finished path." if nav.done else "Stopped (time limit).")
    except KeyboardInterrupt:
        pass
    finally:
        mcu.estop()
        print("E-stop sent.")

        
        if len(path) >= 2 and traj_x:
            px, py = zip(*path)
            plt.figure()
            plt.plot(px, py, '--', label='planned path')
            plt.plot(traj_x, traj_y, label='actual traj')
            plt.plot(traj_x[-1], traj_y[-1], 'o', label='final pos')
            plt.axis('equal'); plt.legend(); plt.title('Ball Picker Coverage')
            plt.xlabel('x [m]'); plt.ylabel('y [m]')
            plt.show()



# Hardware Integration Notes

"""
1) GPS → Local meters:
   - Pick a fixed origin (lat0, lon0). Convert (lat, lon) to local (x,y) via a simple equirectangular approx:
       x = R * cos(lat0) * dlon
       y = R * dlat
     where dlon, dlat in radians, R ≈ 6378137 m.
   - Fuse heading: prefer dual-antenna GNSS yaw if available; else IMU yaw with a light low-pass.
   - Update SensorSuite._sim_update(x, y, yaw, v) with real readings in SensorSuite.read().

2) IMU:
   - Read yaw (heading) in radians; make sure it’s continuous (-pi..pi). Low-pass to reduce jitter.

3) Encoders / Speed:
   - Estimate v from wheel ticks / dt. Optionally blend with GNSS speed.

4) MCU / Motor Control:
   - Map throttle [-1..1] to your controller’s command (PWM or velocity setpoint).
   - Map steer [rad] to steering servo degrees.

5) Safety:
   - Wire a physical E-stop independent of software.
   - Add a “geo-fence” check before sending commands (keep x,y inside range polygon).
   - Add a minimum turning radius to the path generator if towing a trailer (inflate corners).

6) Coverage Polygon (non-rectangular):
   - Replace make_lawnmower_rect() with a polygon-aware generator:
     Create parallel lines every lane_spacing, intersect with polygon, and stitch segments
     in alternating directions (“boustrophedon”). The numeric part is ~30 lines; I can
     add it if you drop me the polygon vertices.

7) Obstacles:
   - For ultrasonics/LiDAR, compute the minimum distance ahead and pass it as soft_obstacle_dist
     to Navigator.step(). For a hard stop, call mcu.estop().
"""

if __name__ == "__main__":
    # If you pass 'sim' it runs the simulator; otherwise it just defines the module.
    if len(sys.argv) == 1 or sys.argv[1] == "sim":
        demo_run()
