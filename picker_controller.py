from controller import Robot, GPS, InertialUnit, Motor
import math

# -----------------------------------------
# Pure Pursuit + PID classes (simplified)
# -----------------------------------------
class PID:
    def __init__(self, kp, ki, kd, limit=1.0):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.i = 0.0; self.prev = 0.0
        self.limit = limit

    def step(self, error, dt):
        self.i += error * dt
        self.i = max(-self.limit, min(self.limit, self.i))
        d = (error - self.prev) / dt
        self.prev = error
        return self.kp*error + self.ki*self.i + self.kd*d


class PurePursuit:
    def __init__(self, wheelbase, steer_limit_rad, lookahead):
        self.wheelbase = wheelbase
        self.steer_limit = steer_limit_rad
        self.lookahead = lookahead
        self.target_idx = 0

    def compute(self, x, y, yaw, path):
        # find target point at lookahead distance
        for i in range(self.target_idx, len(path)):
            dx = path[i][0] - x
            dy = path[i][1] - y
            if math.hypot(dx, dy) >= self.lookahead:
                self.target_idx = i
                break
        tx, ty = path[self.target_idx]
        dx = tx - x
        dy = ty - y

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x_v =  cos_y*dx + sin_y*dy
        y_v = -sin_y*dx + cos_y*dy

        kappa = 2.0 * y_v / (self.lookahead*self.lookahead)
        steer = math.atan(self.wheelbase * kappa)
        return max(-self.steer_limit, min(self.steer_limit, steer))


# -----------------------------------------
# Lawn-mower path generator
# -----------------------------------------
def make_lawnmower_rect(xmin, xmax, ymin, ymax, spacing):
    pts = []
    width = ymax - ymin
    lanes = max(1, int(width/spacing))
    y = ymin + 0.5*spacing
    forward = True
    for _ in range(lanes):
        if forward:
            pts.append((xmin, y))
            pts.append((xmax, y))
        else:
            pts.append((xmax, y))
            pts.append((xmin, y))
        forward = not forward
        y += spacing
    return pts


# -----------------------------------------
# Main Webots controller
# -----------------------------------------

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get sensors
gps = robot.getDevice("gps")
gps.enable(timestep)

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

# Get motors (Pioneer 3-AT)
left_names = ["front left wheel", "rear left wheel"]
right_names = ["front right wheel", "rear right wheel"]

left_m = [robot.getDevice(n) for n in left_names]
right_m = [robot.getDevice(n) for n in right_names]

for m in left_m + right_m:
    m.setPosition(float("inf"))
    m.setVelocity(0)

# Build path
pickup_width = 1.8
overlap = 0.25
spacing = pickup_width * (1 - overlap)
path = make_lawnmower_rect(0, 20, 0, 10, spacing)

# Controllers
wheelbase = 0.32
steer_limit = math.radians(28)
pp = PurePursuit(wheelbase, steer_limit, lookahead=2.0)
pid = PID(kp=0.8, ki=0.2, kd=0.05)

prev_x = 0
prev_y = 0
prev_t = 0
v = 0

while robot.step(timestep) != -1:
    # read pose
    vals = gps.getValues()
    x = vals[0]
    y = vals[2]
    yaw = imu.getRollPitchYaw()[2]

    # compute speed
    t = robot.getTime()
    dt = (t - prev_t) if prev_t > 0 else timestep/1000
    v = math.hypot(x - prev_x, y - prev_y) / dt

    prev_x = x
    prev_y = y
    prev_t = t

    # steering
    steer = pp.compute(x, y, yaw, path)

    # throttle
    desired_v = 1.0
    throttle = pid.step(desired_v - v, dt)
    throttle = max(-1, min(1, throttle))

    # convert to wheel speeds
    L = wheelbase
    wheel_r = 0.095
    omega = desired_v * math.tan(steer) / L

    wl = (desired_v - 0.5*omega*L) / wheel_r
    wr = (desired_v + 0.5*omega*L) / wheel_r

    for m in left_m:
        m.setVelocity(wl)
    for m in right_m:
        m.setVelocity(wr)
