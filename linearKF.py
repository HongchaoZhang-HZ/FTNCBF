import numpy as np

# Define system matrices A, B, and C
def linearKF():
    n = 1
    A = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [3 * n ** 2, 0, 0, 0, 2 * n, 0],
                  [0, 0, 0, -2 * n, 0, 0],
                  [0, 0, -n ** 2, 0, 0, 0]])

    B = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    C1 = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    C3 = np.array([[1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    # Define initial state estimate and covariance
    x_hat = np.zeros((6, 1))  # Initial state estimate
    P = np.eye(6)  # Initial state covariance

    # Define process and measurement noise covariances
    Q = np.eye(6)  # Process noise covariance (adjust as needed)
    R = np.eye(8)  # Measurement noise covariance (adjust as needed)

    # Simulated measurements (replace with your measurements)
    num_time_steps = 100
    measurements = np.random.randn(8, num_time_steps)
    u = np.zeros((3, 1))  # Control input (if any)


    x_hat = np.zeros((6, 1))  # Initial state estimate
    P = np.eye(6)
    # Kalman filter loop
    for k in range(num_time_steps):
        # Prediction step
        x_hat_minus = np.dot(A, x_hat) + np.dot(B, u)  # u is the control input (if any)
        P_minus = np.dot(np.dot(A, P), A.T) + Q

        # Update step
        K1 = np.dot(np.dot(P_minus, C1.T), np.linalg.inv(np.dot(np.dot(C1, P_minus), C1.T) + R))
        x_hat = x_hat_minus + np.dot(K1, (measurements[:, k] - np.dot(C1, x_hat_minus)))
        P = np.dot((np.eye(6) - np.dot(K1, C1)), P_minus)

    x_hat = np.zeros((6, 1))  # Initial state estimate
    P = np.eye(6)
    # Kalman filter loop
    for k in range(num_time_steps):
        # Prediction step
        x_hat_minus = np.dot(A, x_hat) + np.dot(B, u)  # u is the control input (if any)
        P_minus = np.dot(np.dot(A, P), A.T) + Q

        # Update step
        K3 = np.dot(np.dot(P_minus, C3.T), np.linalg.inv(np.dot(np.dot(C3, P_minus), C3.T) + R))
        x_hat = x_hat_minus + np.dot(K3, (measurements[:, k] - np.dot(C3, x_hat_minus)))
        P = np.dot((np.eye(6) - np.dot(K3, C3)), P_minus)

    return K1, K3
