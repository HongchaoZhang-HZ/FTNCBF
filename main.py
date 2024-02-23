import math

import carla
import torch

# from Aug_Percept import *
from kalman_filter import ExtendedKalmanFilter
from Visualization.visualizer import visualizer
from multiprocessing import Queue, Value, Process
from ctypes import c_bool
from car import Car
# import cv2
from util import destroy_queue
from FTEst.ekftest import *
from torch.autograd.functional import hessian
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.controller import VehiclePIDController
from Modules.NCBF import *
from SNCBF_Synth import *
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

from ObsAvoid import ObsAvoid
def waypoint_compute(vehicle, target_angle):
    '''Given control input of target angle, target_angle e.g. math.pi/4
        return waypoint'''
    current_waypoint = vehicle.get_location()
    current_dir_vec = vehicle.get_transform().get_forward_vector()
    v_vec = [current_dir_vec.x, current_dir_vec.y]
    tar_vec_x = 1 * math.cos(math.acos(v_vec[0]) + target_angle)
    tar_vec_y = 1 * math.sin(math.asin(v_vec[0]) + target_angle)
    # off_vec = np.array([off_vec0, off_vec1])

    # nxt_waypoint = current_waypoint
    w_loc = current_waypoint + carla.Location(x=tar_vec_x, y=tar_vec_y)
    # nxt_waypoint = carla.Location(w_loc)
    return w_loc

def d1bdx1(SNCBF, x:float):
    grad_input = torch.tensor(x, requires_grad=True)
    dbdx = torch.autograd.grad(SNCBF.model.forward(grad_input), grad_input)
    return dbdx

def d2bdx2(SNCBF, x):
    grad_input = torch.tensor(x, dtype=torch.float, requires_grad=True)
    hessian_matrix = hessian(SNCBF.model.forward, grad_input).squeeze()
    return hessian_matrix

def multi_SCBF_conditions(self, x):
    cons = []
    gain_list = []
    for SNCBF_idx in range(self.num_SNCBF):
        # Update observation matrix
        obsMatrix = self.FTEst.fault_list.fault_mask_list[SNCBF_idx]
        # Update EKF gain
        EKFGain = self.FTEKF_gain_list[SNCBF_idx]
        # Compute SCBF constraint
        SCBF_cons = self.solo_SCBF_condition(self.SNCBF_list[SNCBF_idx],
                                             x, EKFGain, obsMatrix,
                                             self.gamma_list[SNCBF_idx])
        cons.append(SCBF_cons)

        # Compute Affine Gain
        affine_gain = torch.stack(self.SNCBF_list[SNCBF_idx].get_grad(x)) @ self.gx(x)
        gain_list.append(affine_gain)
    return cons, gain_list

def CBF_based_u(self, x):
    # compute based on self.CBF
    SCBF_cons, affine_gain = self.multi_SCBF_conditions(x)
    cons = tuple()
    for idx in range(self.num_SNCBF):
        SoloCBFCon = lambda u: (affine_gain[idx] @ u).squeeze() + (SCBF_cons[idx]).squeeze()
        SoloOptCBFCon = NonlinearConstraint(SoloCBFCon, 0, np.inf)
        cons = cons + (SoloOptCBFCon,)
    def fcn(u):
        return (u**2).sum()
    # minimize ||u||
    u0 = np.zeros(self.case.CTRLDIM)
    # minimize ||u||
    # constraint: affine_gain @ u + self.SCBF_conditions(x)
    res = minimize(fcn, u0, constraints=SoloOptCBFCon)
    return res

def compute_u():
    def fcn(u):
        return (u**2).sum()
    # minimize ||u||
    u0 = np.array([0])
    res = minimize(fcn, u0)
    return res

def compute_u_scbf(x, SNCBF, K_k):
    def fcn(u):
        return (u**2).sum()
    # minimize ||u||
    dbdx = SNCBF.get_grad(x)[0]
    # stochastic version
    # fx = self.fx(torch.Tensor(x).reshape([1, 3])).numpy()
    fx = (np.eye(3) @ x)
    gx = getB(x[2], 0.1)
    dbdxf = dbdx @ fx
    dbdxg = dbdx @ gx
    c_k = np.array([1,1,1,1,1])
    EKF_term = np.linalg.norm(dbdx @ K_k @ c_k)
    u0 = np.array([0])
    # constraint: affine_gain @ u + self.SCBF_conditions(x)
    hessian = d2bdx2(SNCBF, x)
    second_order_term = 0.1 * K_k.transpose() \
                        @ hessian.numpy() @ K_k * 0.1
    # SoloCBFCon = lambda u: (SNCBF.forward(torch.Tensor(xn))
    #                         - SNCBF.forward(torch.Tensor(x))
    #                         + EKF_term)
    trace_term = second_order_term.trace()
    b_hat = SNCBF(torch.Tensor(x)).detach().numpy()
    # SoloCBFCon = lambda u: (dbdxg[1] * u).squeeze() - 0.001
    # SoloCBFCon = lambda u: (dbdxg[1] * u).squeeze() + dbdxg[0] + dbdxf - EKF_term + trace_term + b_hat - 1
    SoloCBFCon = lambda u: (dbdxg[1] * u).squeeze() + dbdxg[0] + EKF_term + trace_term - 0.042
    SoloOptCBFCon = NonlinearConstraint(SoloCBFCon, 0, np.inf)
    res = minimize(fcn, u0,
                   constraints=SoloOptCBFCon,
                   bounds=[(-math.pi/2, math.pi/2)])
    return res

def compute_u_ftscbf(x_list, SNCBF, K_klist):
    def fcn(u):
        return (u**2).sum()
    # minimize ||u||
    bgamma_list = [0.042, 0.042]

    c_k = np.array([1,1,1,1,1])
    EKF_terms = []
    u0 = np.array([0])
    # constraint: affine_gain @ u + self.SCBF_conditions(x)
    print(x_list[0])
    if x_list[0][1] < -0.5:
        print('here')
    # SoloCBFCon = lambda u: (dbdxg[1] * u).squeeze() - 0.001
    # SoloCBFCon = lambda u: (dbdxg[1] * u).squeeze() + dbdxg[0] + dbdxf - EKF_term + trace_term + b_hat - 1
    SoloCBFCon = lambda u: (-math.sin(x_list[0][2]+u) + x_list[0][1]+0.09)
    SoloOptCBFCon = NonlinearConstraint(SoloCBFCon, 0, np.inf)
    cons = tuple()
    if x_list[0][0] > 0.0:
        cons = cons + (SoloOptCBFCon,)
    for idx in range(len(K_klist)):
        x = x_list[idx]
        dbdx = SNCBF.get_grad(x)[0]
        # stochastic version
        # fx = self.fx(torch.Tensor(x).reshape([1, 3])).numpy()
        fx = (np.eye(3) @ x)
        gx = getB(x[2], 0.1)
        dbdxf = dbdx @ fx
        dbdxg = dbdx @ gx
        EKF_term = np.linalg.norm(dbdx @ K_klist[idx] @ c_k)
        EKF_terms.append(EKF_term)
        hessian = d2bdx2(SNCBF, x)
        K_k = K_klist[idx]
        second_order_term = 0.01 * K_k.transpose() \
                            @ hessian.numpy() @ K_k * 0.01
        # SoloCBFCon = lambda u: (SNCBF.forward(torch.Tensor(xn))
        #                         - SNCBF.forward(torch.Tensor(x))
        #                         + EKF_term)
        trace_term = second_order_term.trace()
        b_hat = SNCBF(torch.Tensor(x)).detach().numpy()
        SoloCBFCon = lambda u: ((dbdxg[1] * u).squeeze() + dbdxg[0] + dbdxf.item() -
                                bgamma_list[idx]*EKF_terms[idx] +
                                trace_term - bgamma_list[idx])
        SoloOptCBFCon = NonlinearConstraint(SoloCBFCon, 0, np.inf)
        cons = cons + (SoloOptCBFCon,)
    # SoloCBFCon = lambda u: (dbdxg[1] * u).squeeze() + dbdxg[0] + EKF_term + trace_term - 0.042
    res = minimize(fcn, u0,
                   constraints=cons,
                   bounds=[(-math.pi/2, math.pi/2)])
    return res


def control_signal(vehicle, tar_angle):
    '''Given target speed and angle,
        return control input for PID controller'''
    tar_waypt = waypoint_compute(vehicle, tar_angle)
    tar_speed = carla.Vector3D(x=1, y=0, z=0)
    return tar_speed, tar_waypt

def main(arg):
    """Main function of the script"""
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(2.0)
    world = client.get_world()
    H1_k = np.array([[1.0, 0, 0],
                     [0, 0, 0],
                     [0, 1.0, 0],
                     [0, 1.0, 0],
                     [0, 0, 1.0]])

    H3_k = np.array([[1.0, 0, 0],
                     [1.0, 0, 0],
                     [0, 1.0, 0],
                     [0, 0, 0],
                     [0, 0, 1.0]])

    H13_k = np.array([[1.0, 0, 0],
                      [0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 0],
                      [0, 0, 1.0]])

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.05

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(arg.filter)[0]

        vehicle_transform = carla.Transform(carla.Location(x=-54.344658, y=137.050995, z=0.600000),
                                            carla.Rotation(yaw=0.352127))

        car = Car(world, client, vehicle_transform)
        dis_pedestrian = 12
        pedestrian_transform_1 = vehicle_transform
        if vehicle_transform.rotation.yaw > 80 and vehicle_transform.rotation.yaw < 100:
            pedestrian_transform_1.location.y += dis_pedestrian  # distance between two cars
        elif vehicle_transform.rotation.yaw > -100 and vehicle_transform.rotation.yaw < -80:
            pedestrian_transform_1.location.y -= dis_pedestrian
        elif vehicle_transform.rotation.yaw > 170 or vehicle_transform.rotation.yaw < -170:
            pedestrian_transform_1.location.x -= dis_pedestrian
        elif vehicle_transform.rotation.yaw > -10 and vehicle_transform.rotation.yaw < 10:
            pedestrian_transform_1.location.x += dis_pedestrian
        pedestrian_transform_1.rotation.yaw -= 90

        walker_bp = blueprint_library.filter("walker.*")[0]
        pedestrian_1 = world.spawn_actor(walker_bp, pedestrian_transform_1)

        # EKF
        ekf = ExtendedKalmanFilter()
        lp = LocalPlanner(car.vehicle)
        lp._vehicle_controller = VehiclePIDController(car.vehicle, args_lateral=lp._args_lateral_dict,
                                                      args_longitudinal=lp._args_longitudinal_dict,
                                                      offset=lp._offset,
                                                      max_throttle=lp._max_throt,
                                                      max_brake=lp._max_brake,
                                                      max_steering=lp._max_steer)

        # Visualizer
        visual_msg_queue = Queue()
        quit = Value(c_bool, False)
        proc = Process(target=visualizer, args=(visual_msg_queue, quit))
        proc.daemon = True
        proc.start()

        # In case Matplotlib is not able to keep up the pace of the growing queue,
        # we have to limit the rate of the items being pushed into the queue
        visual_fps = 10
        last_ts = time.time()

        frame = 0
        dt0 = datetime.now()
        obs = ObsAvoid()
        SNCBF0 = SNCBF_Synth([32, 32], [True, True], obs,
                             sigma=[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
                             nu=[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
                             verbose=True)
        # SNCBF0.model.load_state_dict(torch.load('SNCBF_Obs6.pt'), strict=True)
        SNCBF0.model.load_state_dict(torch.load('Obs_epch5_epsd20_lrate1e-07_batsize32.pt'), strict=True)


        state_estimate_k_minus_1 = np.array([vehicle_transform.location.x,
                                             vehicle_transform.location.y,
                                             vehicle_transform.rotation.yaw])
        state_estimate_k1_minus_1 = state_estimate_k_minus_1
        state_estimate_k3_minus_1 = state_estimate_k_minus_1
        obs_vector_z_k = np.array([vehicle_transform.location.x,
                                   vehicle_transform.location.x,
                                   vehicle_transform.location.y,
                                   vehicle_transform.location.y,
                                   vehicle_transform.rotation.yaw])
        control_vector_k_minus_1 = np.array([10.0, 0.0])
        P_k_minus_1 = np.array([[0.1, 0, 0],
                                [0, 0.1, 0],
                                [0, 0, 0.1]])
        P_k1_minus_1 = P_k_minus_1
        P_k3_minus_1 = P_k_minus_1
        dk = 1.0/visual_fps
        for iteration in range(10000):
            world.tick()
            frame = world.get_snapshot().frame

            # Get sensor readings
            sensors = car.get_sensor_readings(frame)

            # get image
            if sensors['image'] is not None:
                image = sensors['image']
                cv2.imshow('image', image)
                cv2.waitKey(1)

                # EKF initialization
            # Don't run anything else before EKF is initialized
            if not ekf.is_initialized():
                if sensors['gnss'] is not None:
                    ekf.initialize_with_gnss(sensors['gnss'])

                continue

            # EKF prediction
            if sensors['imu'] is not None:
                ekf.predict_state_with_imu(sensors['imu'])

            # EKF correction
            if sensors['gnss'] is not None:
                ekf.correct_state_with_gnss(sensors['gnss'])

            # Limit the visualization frame-rate
            if time.time() - last_ts < 1. / visual_fps:
                continue

            # timestamp for inserting a new item into the queue
            last_ts = time.time()

            # visual message
            visual_msg = dict()

            # Get ground truth vehicle location
            gt_location = car.get_location()
            visual_msg['gt_traj'] = [gt_location.x, gt_location.y, gt_location.z]  # round(x, 1)

            # Get estimated location
            visual_msg['est_traj'] = ekf.get_location()

            # Get imu reading
            if sensors['imu'] is not None:
                imu = sensors['imu']
                accelero = imu.accelerometer
                gyroscop = imu.gyroscope
                visual_msg['imu'] = [accelero.x, accelero.y, accelero.z,
                                     gyroscop.x, gyroscop.y, gyroscop.z]

            # Get gps reading
            if sensors['gnss'] is not None:
                gnss = sensors['gnss']
                visual_msg['gnss'] = [gnss.x, gnss.y, gnss.z]
                if iteration <=500:
                    obs_vector_z_k = np.array([gnss.x, gnss.x, gnss.y, gnss.y, gnss.z])
                else:
                    # obs_vector_z_k = np.array([gnss.x, gnss.x, gnss.y, gnss.y, gnss.z])
                    obs_vector_z_k = np.array([gnss.x, gnss.x, gnss.y, gnss.y + 2, gnss.z])

            visual_msg_queue.put(visual_msg)

            optimal_state_estimate_k1, covariance_estimate_k1, K_k1 = ekf0test(
                obs_vector_z_k,  # Most recent sensor measurement
                state_estimate_k1_minus_1,  # Our most recent estimate of the state
                control_vector_k_minus_1,  # Our most recent control input
                P_k1_minus_1,  # Our most recent state covariance matrix
                H1_k,
                dk)  # Time interval

            optimal_state_estimate_k3, covariance_estimate_k3, K_k3 = ekf0test(
                obs_vector_z_k,  # Most recent sensor measurement
                state_estimate_k3_minus_1,  # Our most recent estimate of the state
                control_vector_k_minus_1,  # Our most recent control input
                P_k3_minus_1,  # Our most recent state covariance matrix
                H13_k,
                dk)  # Time interval

            # Get ready for the next timestep by updating the variable values
            if iteration >= 500:
                hat1x0 = (pedestrian_transform_1.location.x - optimal_state_estimate_k1[0])/10
                hat1x1 = (pedestrian_transform_1.location.y - optimal_state_estimate_k1[1])/10
                hat1phi = optimal_state_estimate_k1[2]
                # u = -math.sin(phi)*3*(x0*math.sin(phi)+x1*math.cos(phi))/(x0**2+x1**2)
                xhat1 = np.array([hat1x0,hat1x1,hat1phi])

                hat3x0 = (pedestrian_transform_1.location.x - optimal_state_estimate_k3[0]) / 10
                hat3x1 = (pedestrian_transform_1.location.y - optimal_state_estimate_k3[1]) / 10
                hat3phi = optimal_state_estimate_k3[2]
                # u = -math.sin(phi)*3*(x0*math.sin(phi)+x1*math.cos(phi))/(x0**2+x1**2)
                xhat3 = np.array([hat3x0, hat3x1, hat3phi])
                x_list = [xhat1, xhat3]
                K_klist = [K_k1, K_k3]
                print(np.linalg.norm(xhat1-xhat3))
                # if np.linalg.norm(xhat1-xhat3)>=0.1:
                #     x_list = [xhat3]
                #     K_klist = [K_k3]
                res = compute_u_ftscbf(x_list, SNCBF0, K_klist)
                u = res.x
                control_vector_k_minus_1 = np.array([10.0, u[0]])
                # print(SNCBF0(torch.Tensor(x)))
                # print(u)
            state_estimate_k1_minus_1 = optimal_state_estimate_k1
            P_k1_minus_1 = covariance_estimate_k1
            state_estimate_k3_minus_1 = optimal_state_estimate_k3
            P_k3_minus_1 = covariance_estimate_k3
            # print(optimal_state_estimate_k, obs_vector_z_k)
            # if iteration >= 480 and iteration <500:
            #     ctrl = car.vehicle.get_control()
            #     ctrl.throttle = 0.2
            #     car.vehicle.apply_control(ctrl)

            if iteration >= 500:
                # car.vehicle.set_velocity(carla.Vector3D(x=10,y=0,z=0))
                tar_waypt = waypoint_compute(car.vehicle, u)
                wp = lp._map.get_waypoint(tar_waypt)
                ctrl = lp._vehicle_controller.run_step(10,wp)
                if x_list[0][0] > 0:
                    ctrl.steer = u[0]
                    car.vehicle.apply_control(ctrl)
                else:
                    car.vehicle.set_autopilot(True)
            # ctrl = car.vehicle.get_control()
            # ctrl.steer = 0.5
            # ctrl.throttle = 0.2

            # print(car.vehicle.get_velocity().x)

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        print('Exiting visualizer')
        quit.value = True
        destroy_queue(visual_msg_queue)

        print('destroying the car object')
        car.destroy()
        pedestrian_1.destroy()

        print('done')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_true',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=500000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
