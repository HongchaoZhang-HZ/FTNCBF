from EKF import *
from SensorFaults import *


class RdObsEKF(RobotEKF):
    def __init__(self, sensors: SensorSet,
                 dt, wheelbase, std_vel, std_steer,
                 std_range, std_bearing, verbose=False):
        self.verbose = verbose
        self.sensor_list = sensors
        self.num_sensors = self.sensor_list.num_sensors
        self.obsMatrix = self.sensor_list.obs_matrix
        self.obsVector = np.linalg.norm(self.obsMatrix, axis=1).reshape([self.num_sensors, 1])
        # TODO: redundancy check
        # TODO: observability check
        EKF.__init__(self, 3, self.num_sensors, 1)
        self.dt = dt
        self.wheelbase = wheelbase
        self.std_vel = std_vel
        self.std_steer = std_steer
        self.std_range = std_range
        self.std_bearing = std_bearing

        a, x, y, v, w, theta, time = symbols(
            'a, x, y, v, w, theta, t')
        d = v * time
        beta = (d / w) * sympy.tan(a)
        r = w / sympy.tan(a)

        self.fxu = Matrix(
            [[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
             [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
             [theta + beta]])

        self.F_j = self.fxu.jacobian(Matrix([x, y, theta]))
        self.V_j = self.fxu.jacobian(Matrix([a]))

        # save dictionary and it's variables for later use
        self.subs = {x: 0, y: 0, v: 0, a: 0,
                     time: dt, w: wheelbase, theta: 0}
        self.x_x, self.x_y, = x, y
        self.v, self.a, self.theta = v, a, theta

    def predict(self, u):
        self.x = self.move(self.x, u, self.dt)
        self.subs[self.x_x] = self.x[0, 0]
        self.subs[self.x_y] = self.x[1, 0]

        self.subs[self.theta] = self.x[2, 0]
        self.subs[self.v] = 1
        self.subs[self.a] = u[0]

        F = np.array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = np.array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = np.array([[self.std_steer ** 2]])

        self.P = F @ self.P @ F.T + V @ M @ V.T

    def obs_change(self, new_obsMatrix):
        self.obsMatrix = new_obsMatrix

    def move(self, x, u, dt):
        hdg = x[2, 0]
        vel = 1
        steering_angle = u[0]
        dist = vel * dt

        if abs(steering_angle) > 0.001:  # is robot turning?
            beta = (dist / self.wheelbase) * tan(steering_angle)
            r = self.wheelbase / tan(steering_angle)  # radius

            dx = np.array([[-r * sin(hdg) + r * sin(hdg + beta)],
                           [r * cos(hdg) - r * cos(hdg + beta)],
                           [beta]])
        else:  # moving in straight line
            dx = np.array([[dist * cos(hdg)],
                           [dist * sin(hdg)],
                           [0]])
        return x + dx

    def H_of(self, x, landmark_pos):
        """ compute Jacobian of H matrix where h(x) computes
        the range and bearing to a landmark for state x """

        px = landmark_pos[0]
        py = landmark_pos[1]
        hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
        dist = sqrt(hyp)

        # H = np.array(
        #     [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
        #      [ (py - x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1]])
        H = np.array(
            [[-(px - x[0, 0]) / dist, 0, 0],
             [0, -(py - x[1, 0]) / dist, 0],
             [0, -(py - x[1, 0]) / dist, 0],
             [(py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1],
             [(py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1]])
        return H * self.obsMatrix

    def Hx(self, x, landmark_pos):
        """ takes a state variable and returns the measurement
        that would correspond to that state.
        """
        px = landmark_pos[0]
        py = landmark_pos[1]
        # dist = sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

        # Hx = np.array([[dist],
        #             [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
        Hx = np.array([[sqrt((px - x[0, 0]) ** 2)],
                       [sqrt((py - x[1, 0]) ** 2)],
                       [sqrt((py - x[1, 0]) ** 2)],
                       [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]],
                       [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
        return Hx * self.obsVector

    def residual(self, a, b):
        """ compute residual (a-b) between measurements containing
        [range, bearing]. Bearing is normalized to [-pi, pi)"""
        y = a - b
        y[3] = y[3] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[3] > np.pi:  # move to [-pi, pi)
            y[3] -= 2 * np.pi
        y[4] = y[4] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[4] > np.pi:  # move to [-pi, pi)
            y[4] -= 2 * np.pi
        return y

    def z_landmark(self, lmark, sim_pos, std_rng, std_brg):
        x, y = sim_pos[0, 0], sim_pos[1, 0]
        d = np.sqrt((lmark[0] - x) ** 2 + (lmark[1] - y) ** 2)
        a = atan2(lmark[1] - y, lmark[0] - x) - sim_pos[2, 0]
        # z = np.array([[d + np.random.randn() * std_rng],
        #               [d + np.random.randn() * std_rng],
        #               [a + np.random.randn() * std_brg]])
        z = np.array([[(lmark[0] - x) + np.random.randn() * std_rng],
                      [(lmark[1] - y) + np.random.randn() * std_rng],
                      [(lmark[1] - y) + np.random.randn() * std_rng],
                      [a + np.random.randn() * std_brg],
                      [a + np.random.randn() * std_brg]])
        return z

    def ekf_update(self, z, landmark):
        self.update(z, HJacobian=self.H_of, Hx=self.Hx,
                    residual=self.residual,
                    args=(landmark), hx_args=(landmark))

    def run_localization(self, landmarks,
                         step=10, ellipse_step=20, ylim=None):
        # TODO: get P being a function of x and K as a function of x
        self.x = np.array([[2, 6, .3]]).T  # x, y, steer angle
        self.P = np.diag([.1, .1, .1])
        self.R = np.diag([self.std_range ** 2,
                          self.std_range ** 2, self.std_range ** 2,
                          self.std_bearing ** 2, self.std_bearing ** 2])

        sim_pos = self.x.copy()  # simulated position
        # steering command (vel, steering angle radians)
        u = np.array([.01])

        if self.verbose:
            plt.figure()
            plt.scatter(landmarks[:, 0], landmarks[:, 1],
                        marker='s', s=60)

        track = []
        for i in range(200):
            sim_pos = self.move(sim_pos, u, dt / 10.)  # simulate robot
            track.append(sim_pos)
            # print(ekf.K)

            if i % step == 0:
                self.predict(u=u)

                if i % ellipse_step == 0 and self.verbose:
                    plot_covariance_ellipse(
                        (self.x[0, 0], self.x[1, 0]), self.P[0:2, 0:2],
                        std=6, facecolor='k', alpha=0.3)

                x, y = sim_pos[0, 0], sim_pos[1, 0]
                for lmark in landmarks:
                    z = self.z_landmark(lmark, sim_pos,
                                        self.std_range, self.std_bearing)
                    self.ekf_update(z, lmark)

                if i % ellipse_step == 0 and self.verbose:
                    plot_covariance_ellipse(
                        (self.x[0, 0], self.x[1, 0]), self.P[0:2, 0:2],
                        std=6, facecolor='g', alpha=0.8)
        track = np.array(track)
        if self.verbose:
            plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
            plt.axis('equal')
            plt.title("EKF Robot localization")
            if ylim is not None: plt.ylim(*ylim)
            plt.show()

# dt = 1.0
# sensor_list = SensorSet([0, 1, 1, 2, 2], [0.001, 0.002, 0.0015, 0.001, 0.01])
# fault_list = FaultPattern(sensor_list,
#                           fault_target=[[1], [2, 3]],
#                           fault_value=[[0.1], [0.15, 2]])
# landmarks = np.array([[5, 5, 10, 10, 0.5], [10, 10, 5, 5, 0.5], [15, 15, 15, 15, 0.5]])
# ekf = RdObsEKF(sensor_list, dt, wheelbase=0.5, std_vel=0.1,
#                std_steer=np.radians(1), std_range=0.3, std_bearing=0.1)
# ekf.run_localization(landmarks)
# print('Final K:', ekf.K)
