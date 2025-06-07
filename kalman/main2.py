import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilterGPSCVAReorder:
    def __init__(self, jerk_var=0.05, initial_cov=100):
        """
        Constant-Acceleration Kalman Filter with reordered state:
        state x = [p_lat, v_lat, a_lat, p_lon, v_lon, a_lon]

        Parameters:
        -----------
        jerk_var : float
            Process noise parameter for jerk (rate of change of acceleration)
        initial_cov : float
            Initial covariance for state variables
        """
        self.jerk_var = jerk_var
        self.initial_cov = initial_cov
        self.x = None
        self.P = None
        self.last_timestamp = None

    def initialize(self, p_lat0, v_lat0, p_lon0, v_lon0, a_lat0=0.0, a_lon0=0.0):
        """
        Initialize the 6-element state and covariance.

        Parameters:
        -----------
        p_lat0, p_lon0 : float
            Initial position (latitude, longitude)
        v_lat0, v_lon0 : float
            Initial velocity (latitude, longitude) in degrees/second
        a_lat0, a_lon0 : float
            Initial acceleration (latitude, longitude) in degrees/second²
        """
        # [position, velocity, acceleration] for lat and lon
        self.x = np.array([p_lat0, v_lat0, a_lat0, p_lon0, v_lon0, a_lon0], dtype=float)

        # Initialize covariance matrix with different uncertainties for pos/vel/acc
        self.P = np.diag([
            self.initial_cov,  # position lat
            self.initial_cov * 10,  # velocity lat
            self.initial_cov * 20,  # acceleration lat
            self.initial_cov,  # position lon
            self.initial_cov * 10,  # velocity lon
            self.initial_cov * 20  # acceleration lon
        ])

    def predict(self, dt):
        """
        Predict step using constant acceleration model:
          p_{k+1} = p_k + v_k*dt + 0.5*a_k*dt^2
          v_{k+1} = v_k + a_k*dt
          a_{k+1} = a_k

        Parameters:
        -----------
        dt : float
            Time step in seconds
        """
        # Limit dt to reasonable values to prevent numerical issues
        dt = max(0.01, min(dt, 10.0))

        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        # State transition matrix
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
            [dt4 / 24, dt3 / 6, dt2 / 2, 0, 0, 0],
            [dt3 / 6, dt2 / 2, dt, 0, 0, 0],
            [dt2 / 2, dt, 1, 0, 0, 0],
            [0, 0, 0, dt4 / 24, dt3 / 6, dt2 / 2],
            [0, 0, 0, dt3 / 6, dt2 / 2, dt],
            [0, 0, 0, dt2 / 2, dt, 1]
        ], dtype=float) * self.jerk_var ** 2

        # predict state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R):
        """
        Update step with measurement z and measurement noise R.

        Parameters:
        -----------
        z : numpy.ndarray
            Measurement vector [p_lat, v_lat, a_lat, p_lon, v_lon, a_lon]
        R : numpy.ndarray
            Measurement noise covariance matrix (6x6)
        """
        H = np.eye(6)  # Measurement matrix (identity for full state measurement)
        y = z - H @ self.x  # Innovation/residual

        # Innovation covariance
        S = H @ self.P @ H.T + R

        try:
            # Kalman Gain
            K = self.P @ H.T @ np.linalg.inv(S)

            # State update equation
            self.x = self.x + K @ y

            # Joseph form covariance update (more numerically stable)
            I = np.eye(6)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        except np.linalg.LinAlgError:
            # Fallback if matrix inversion fails
            print("Warning: Matrix inversion failed in Kalman update")

    def smooth_acceleration(self, a_lat, a_lon, prev_a_lat, prev_a_lon, alpha=0.3):
        """
        Apply exponential smoothing to acceleration estimates to reduce noise.

        Parameters:
        -----------
        a_lat, a_lon : float
            Current acceleration estimates
        prev_a_lat, prev_a_lon : float
            Previous acceleration estimates
        alpha : float
            Smoothing factor (0-1), lower values mean more smoothing

        Returns:
        --------
        float, float
            Smoothed acceleration estimates
        """
        if prev_a_lat is None or prev_a_lon is None:
            return a_lat, a_lon

        a_lat_smooth = alpha * a_lat + (1 - alpha) * prev_a_lat
        a_lon_smooth = alpha * a_lon + (1 - alpha) * prev_a_lon

        return a_lat_smooth, a_lon_smooth

    def filter(self, df):
        """
        Apply filter to DataFrame with columns:
        fixtime, latitude, longitude, accuracy, speed, course.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with GPS measurements

        Returns:
        --------
        numpy.ndarray, numpy.ndarray
            Filtered latitude and longitude arrays
        """
        df = df.copy()

        # Convert fixtime to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['fixtime']):
            df['fixtime'] = pd.to_datetime(df['fixtime'])

        # Sort by timestamp to ensure proper time sequence
        df = df.sort_values('fixtime').reset_index(drop=True)

        # Remove duplicates based on timestamp
        df = df.drop_duplicates(subset=['fixtime']).reset_index(drop=True)

        # Convert speed from m/s to deg/s and compute velocity components
        v_lat = []
        v_lon = []
        for spd, crs, lat in zip(df['speed'], df['course'], df['latitude']):
            rad = np.deg2rad(crs)
            vn = spd * np.cos(rad)  # North component in m/s
            ve = spd * np.sin(rad)  # East component in m/s

            # Convert to deg/s - 111000 meters per degree of latitude
            # For longitude, need to account for latitude
            v_lat.append(vn / 111000.0)
            v_lon.append(ve / (111000.0 * np.cos(np.deg2rad(lat))))

        df['v_lat'] = v_lat
        df['v_lon'] = v_lon

        # Initialize with first fix and measured velocity
        p_lat0 = df.loc[0, 'latitude']
        p_lon0 = df.loc[0, 'longitude']
        v_lat0 = df.loc[0, 'v_lat']
        v_lon0 = df.loc[0, 'v_lon']

        self.initialize(p_lat0, v_lat0, p_lon0, v_lon0)
        self.last_timestamp = df.loc[0, 'fixtime']

        lat_f, lon_f = [], []
        prev_v_lat, prev_v_lon = None, None
        prev_a_lat, prev_a_lon = None, None

        for i, row in df.iterrows():
            # Calculate time step
            current_time = row['fixtime']
            dt = (current_time - self.last_timestamp).total_seconds()

            # Skip if time difference is too small (duplicate or very close measurements)
            if dt < 0.01 and i > 0:
                # Use previous filtered position
                lat_f.append(lat_f[-1])
                lon_f.append(lon_f[-1])
                continue

            # Reasonable default for first point
            if dt <= 0:
                dt = 1.0

            # Prediction step
            self.predict(dt)

            # Measure acceleration from Δv/Δt
            if prev_v_lat is not None and dt > 0:
                a_lat = (row['v_lat'] - prev_v_lat) / dt
                a_lon = (row['v_lon'] - prev_v_lon) / dt

                # Apply smoothing to acceleration estimates
                a_lat, a_lon = self.smooth_acceleration(a_lat, a_lon, prev_a_lat, prev_a_lon)
            else:
                a_lat = a_lon = 0.0

            # Measurement noise based on GPS accuracy
            # Convert meters to degrees and scale appropriately
            sigma_p = max(row['accuracy'], 1.0) / 111000.0  # Position noise (deg)

            # Velocity and acceleration noise scaled based on position accuracy
            # Higher position uncertainty means higher velocity and acceleration uncertainty
            sigma_v = sigma_p / max(dt, 0.1)
            sigma_a = sigma_v / max(dt, 0.1)

            # Build measurement noise covariance matrix
            # Higher uncertainty for velocity and acceleration measurements
            R = np.diag([
                sigma_p ** 2,  # p_lat
                (sigma_v * 1.5) ** 2,  # v_lat (1.5x higher noise)
                (sigma_a * 3.0) ** 2,  # a_lat (3x higher noise)
                sigma_p ** 2,  # p_lon
                (sigma_v * 1.5) ** 2,  # v_lon
                (sigma_a * 3.0) ** 2  # a_lon
            ])

            # Handle very slow movement - if speed is below threshold,
            # don't trust course/direction measurements as much
            if row['speed'] < 0.5:  # below 0.5 m/s
                R[1, 1] *= 5  # Increase v_lat uncertainty
                R[4, 4] *= 5  # Increase v_lon uncertainty
                R[2, 2] *= 5  # Increase a_lat uncertainty
                R[5, 5] *= 5  # Increase a_lon uncertainty

            # Measurement vector z in reordered state order
            z = np.array([
                row['latitude'],  # p_lat
                row['v_lat'],  # v_lat
                a_lat,  # a_lat
                row['longitude'],  # p_lon
                row['v_lon'],  # v_lon
                a_lon  # a_lon
            ], dtype=float)

            # Update step
            self.update(z, R)

            # Store filtered positions
            lat_f.append(self.x[0])
            lon_f.append(self.x[3])

            # Update previous values
            prev_v_lat, prev_v_lon = row['v_lat'], row['v_lon']
            prev_a_lat, prev_a_lon = a_lat, a_lon
            self.last_timestamp = current_time

        return np.array(lat_f), np.array(lon_f)


def plot_results(df, show_accel=False):
    """
    Plot filtered vs raw GPS data

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with original and filtered GPS data
    show_accel : bool
        Whether to plot acceleration data
    """
    # Position plots
    plt.figure(figsize=(12, 8))

    # Latitude plot
    plt.subplot(2, 2, 1)
    plt.plot(df['fixtime'], df['latitude'], 'b.', label='Raw Latitude', alpha=0.7)
    plt.plot(df['fixtime'], df['lat_kf'], 'r-', label='Filtered Latitude')
    plt.xlabel('Time')
    plt.ylabel('Latitude')
    plt.title('Latitude: Raw vs Filtered')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Longitude plot
    plt.subplot(2, 2, 2)
    plt.plot(df['fixtime'], df['longitude'], 'b.', label='Raw Longitude', alpha=0.7)
    plt.plot(df['fixtime'], df['lon_kf'], 'r-', label='Filtered Longitude')
    plt.xlabel('Time')
    plt.ylabel('Longitude')
    plt.title('Longitude: Raw vs Filtered')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2D track plot
    plt.subplot(2, 2, 3)
    plt.plot(df['longitude'], df['latitude'], 'b.', label='Raw Track', alpha=0.7)
    plt.plot(df['lon_kf'], df['lat_kf'], 'r-', label='Filtered Track')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('2D GPS Track')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Speed plot if available
    plt.subplot(2, 2, 4)
    if 'speed' in df.columns:
        plt.plot(df['fixtime'], df['speed'], 'g-', label='Speed (m/s)')
        plt.xlabel('Time')
        plt.ylabel('Speed (m/s)')
        plt.title('Speed')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Optional acceleration plots
    if show_accel and 'v_lat' in df.columns and len(df) > 1:
        plt.figure(figsize=(12, 6))

        # Calculate accelerations
        df['a_lat'] = df['v_lat'].diff() / df['fixtime'].diff().dt.total_seconds()
        df['a_lon'] = df['v_lon'].diff() / df['fixtime'].diff().dt.total_seconds()

        plt.subplot(1, 2, 1)
        plt.plot(df['fixtime'][1:], df['a_lat'][1:], 'b-', label='Latitude Acceleration')
        plt.xlabel('Time')
        plt.ylabel('Acceleration (deg/s²)')
        plt.title('Latitude Acceleration')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(df['fixtime'][1:], df['a_lon'][1:], 'r-', label='Longitude Acceleration')
        plt.xlabel('Time')
        plt.ylabel('Acceleration (deg/s²)')
        plt.title('Longitude Acceleration')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('newtrack_deduped.csv')


    # Convert fixtime to datetime
    df['fixtime'] = pd.to_datetime(df['fixtime'])

    # Apply the improved Kalman filter
    kf = KalmanFilterGPSCVAReorder(jerk_var=0.02, initial_cov=50)
    df['lat_kf'], df['lon_kf'] = kf.filter(df)

    # Plot results
    plot_results(df, show_accel=True)
