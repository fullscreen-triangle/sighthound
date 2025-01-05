from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd


def dynamic_kalman_filter(data, process_noise=1e-5, measurement_noise=1e-1):
    metrics = [col for col in data.columns if col not in ['timestamp', 'latitude', 'longitude']]
    dim_x = 4 + len(metrics)  # [x, y, dx, dy, metrics...]
    dim_z = 2  # Measurement dimensions: [latitude, longitude]

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = np.eye(dim_x)
    for i in range(2, 4):
        kf.F[i, i - 2] = 1
    kf.H = np.zeros((dim_z, dim_x))
    kf.H[0, 0] = 1
    kf.H[1, 1] = 1

    kf.P *= 1000
    kf.R *= measurement_noise
    kf.Q *= process_noise

    results = []
    for _, row in data.iterrows():
        z = np.array([row['latitude'], row['longitude']])
        kf.predict()
        kf.update(z)

        state = list(kf.x[:4])
        for i, metric in enumerate(metrics):
            state.append(kf.x[4 + i])
        results.append(state)

    column_names = ['latitude', 'longitude', 'dx', 'dy'] + metrics
    smoothed_data = pd.DataFrame(results, columns=column_names)

    for col in smoothed_data.columns:
        if col in data.columns:
            data[col] = smoothed_data[col]

    return data
