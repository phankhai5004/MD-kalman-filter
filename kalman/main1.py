import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilterGPSCVAReorder:
    def __init__(self, jerk_var=0.2):
        """
        Constant-Acceleration Kalman Filter with reordered state:
        state x = [p_lat, v_lat, a_lat, p_lon, v_lon, a_lon]
        """
        self.jerk_var = jerk_var
        self.x = None
        self.P = None

    def initialize(self, p_lat0, v_lat0, p_lon0, v_lon0, cov0=500):
        """
        Initialize the 6-element state and covariance.
        """
        # [position, velocity, acceleration] for lat and lon
        self.x = np.array([p_lat0, v_lat0, 0.0, p_lon0, v_lon0, 0.0], dtype=float)
        self.P = np.eye(6) * cov0

    def predict(self, dt):
        """
        Predict step using constant acceleration model:
          p _{k+1} = p_k + v_k*dt + 0.5*a_k*dt^2
          v_{k+1} = v_k + a_k*dt
          a_{k+1} = a_k
        """
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        F = np.array([
            [1, dt, 0.5 * dt2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Process noise Q: jerk variance on acceleration states
        Q = np.array([
            [dt4 / 4, dt3 / 2, dt2 / 2, 0, 0, 0],
            [dt3 / 2, dt2, dt, 0, 0, 0],
            [dt2 / 2, dt, 1, 0, 0, 0],
            [0, 0, 0, dt4 / 4, dt3 / 2, dt2 / 2],
            [0, 0, 0, dt3 / 2, dt2, dt],
            [0, 0, 0, dt2 / 2, dt, 1]
        ], dtype=float) * self.jerk_var ** 2

        # predict state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R):
        """
        Update step with full-state measurement z (6x1).
        """
        H = np.eye(6)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def filter(self, df):
        """
        Apply filter to DataFrame with columns:
        fixtime, latitude, longitude, accuracy, speed, course.
        Returns filtered position arrays.
        """
        df = df.copy()
        df['fixtime'] = pd.to_datetime(df['fixtime'])

        # compute measured velocities (deg/s)
        v_lat = []
        v_lon = []
        for spd, crs, lat in zip(df['speed'], df['course'], df['latitude']):
            rad = np.deg2rad(crs)
            vn = spd * np.cos(rad)
            ve = spd * np.sin(rad)
            v_lat.append(vn / 111000.0)
            v_lon.append(ve / (111000.0 * np.cos(np.deg2rad(lat))))
        df['v_lat'] = v_lat
        df['v_lon'] = v_lon

        # initialize with first fix and first measured velocity
        p_lat0 = df.loc[0, 'latitude']
        p_lon0 = df.loc[0, 'longitude']
        v_lat0 = df.loc[0, 'v_lat']
        v_lon0 = df.loc[0, 'v_lon']
        self.initialize(p_lat0, v_lat0, p_lon0, v_lon0, cov0=500)

        lat_f, lon_f = [], []
        prev_v_lat, prev_v_lon = None, None

        for i, row in df.iterrows():
            # time step
            if i == 0:
                dt = 1.0
            else:
                dt = (row['fixtime'] - df.loc[i - 1, 'fixtime']).total_seconds()

            # prediction
            self.predict(dt)

            # measure acceleration from Δv/Δt
            if prev_v_lat is not None:
                a_lat = (row['v_lat'] - prev_v_lat) / dt
                a_lon = (row['v_lon'] - prev_v_lon) / dt
            else:
                a_lat = a_lon = 0.0

            # measurement noise: position, velocity, acceleration
            sigma_p = row['accuracy'] / 111000.0
            sigma_v = sigma_p / dt if dt > 0 else sigma_p
            sigma_a = sigma_v / dt if dt > 0 else sigma_v
            R = np.diag([
                sigma_p ** 2,  # p_lat
                sigma_v ** 2,  # v_lat
                sigma_a ** 2,  # a_lat
                sigma_p ** 2,  # p_lon
                sigma_v ** 2,  # v_lon
                sigma_a ** 2,  # a_lon
            ])

            # measurement vector z in reordered state order
            z = np.array([
                row['latitude'],  # p_lat
                row['v_lat'],  # v_lat
                a_lat,  # a_lat
                row['longitude'],  # p_lon
                row['v_lon'],  # v_lon
                a_lon  # a_lon
            ], dtype=float)

            # update
            self.update(z, R)

            # store filtered positions
            lat_f.append(self.x[0])
            lon_f.append(self.x[3])

            prev_v_lat, prev_v_lon = row['v_lat'], row['v_lon']

        return np.array(lat_f), np.array(lon_f)


# Load deduplicated CSV
df = pd.read_csv('newtrack_deduped.csv')

# Apply the reordered CVA filter
kf = KalmanFilterGPSCVAReorder(jerk_var=0.01)
df['lat_kf'], df['lon_kf'] = kf.filter(df)

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(df['fixtime'], df['latitude'], label='Raw Latitude')
plt.plot(df['fixtime'], df['lat_kf'], label='Filtered Latitude')
plt.xlabel('Time')
plt.ylabel('Latitude')
plt.title('Latitude: CVA Reordered State')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(df['fixtime'], df['longitude'], label='Raw Longitude')
plt.plot(df['fixtime'], df['lon_kf'], label='Filtered Longitude')
plt.xlabel('Time')
plt.ylabel('Longitude')
plt.title('Longitude: CVA Reordered State')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(df['longitude'], df['latitude'], '.', label='Raw Track')
plt.plot(df['lon_kf'], df['lat_kf'], '.', label='Filtered Track')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('2D GPS Track: CVA Reordered State')
plt.legend()
plt.tight_layout()
plt.show()
