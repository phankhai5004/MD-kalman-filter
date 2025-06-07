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

        # Kalman Gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update equation
        self.x = self.x + K @ y

        # Covariance Update equation
        I = np.eye(6)
        t = (I - K @ H)
        self.P = t @ self.P @ t.T + K @ R @ K.T

    def filter(self, df):
        """
        Apply filter to DataFrame with columns:
        fixtime, latitude, longitude, accuracy, speed, course.
        Returns filtered position arrays.
        """
        df = df.copy()
        df['fixtime'] = pd.to_datetime(df['fixtime'])

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

            # Measurement noise based on GPS accuracy
            # Convert meters to degrees and scale appropriately
            sigma_p = row['accuracy'] / 111000.0  # Position noise (deg)

            # Velocity and acceleration noise scaled based on position accuracy
            # Higher position uncertainty means higher velocity and acceleration uncertainty
            sigma_v = sigma_p / dt
            sigma_a = sigma_v / dt

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


            # # measurement noise: position, velocity, acceleration in rad
            # sigma_p = row['accuracy'] / 111000.0
            # sigma_v = sigma_p / dt if dt > 0 else sigma_p
            # sigma_a = sigma_v / dt if dt > 0 else sigma_v
            # R = np.diag([
            #     sigma_p ** 2,  # p_lat
            #     sigma_v ** 2,  # v_lat
            #     sigma_a ** 2,  # a_lat
            #     sigma_p ** 2,  # p_lon
            #     sigma_v ** 2,  # v_lon
            #     sigma_a ** 2,  # a_lon
            # ])

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

def plot_results(df):
    """
    Plot filtered vs raw GPS data in separate square graphs
    """
    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(df['fixtime'], df['latitude'], label='Raw Latitude')
    plt.plot(df['fixtime'], df['lat_kf'], label='Filtered Latitude')
    plt.xlabel('Time')
    plt.ylabel('Latitude')
    plt.title('Latitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(df['fixtime'], df['longitude'], label='Raw Longitude')
    plt.plot(df['fixtime'], df['lon_kf'], label='Filtered Longitude')
    plt.xlabel('Time')
    plt.ylabel('Longitude')
    plt.title('Longitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(df['longitude'], df['latitude'], '-', label='Raw Track')
    plt.plot(df['lon_kf'], df['lat_kf'], '-', label='Filtered Track')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Kalman Filter')
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_metrics(df):
    """
    Calculate and display filter performance metrics.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with raw and filtered GPS data
    """
    print("\n" + "=" * 60)
    print("          KALMAN FILTER PERFORMANCE METRICS")
    print("=" * 60)

    # 1. Standard Deviation Comparison - Raw vs Filtered
    lat_raw_std = np.std(df['latitude'])
    lon_raw_std = np.std(df['longitude'])
    lat_filtered_std = np.std(df['lat_kf'])
    lon_filtered_std = np.std(df['lon_kf'])

    # Combined STD calculations
    combined_raw_std = np.sqrt(lat_raw_std ** 2 + lon_raw_std ** 2)
    combined_filtered_std = np.sqrt(lat_filtered_std ** 2 + lon_filtered_std ** 2)

    print(f"\n📍 STANDARD DEVIATION COMPARISON (Raw vs Filtered):")
    print(f"   {'Metric':<15} {'Raw':<15} {'Filtered':<15} {'Improvement':<15}")
    print(f"   {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15}")

    # Degrees
    lat_std_improvement = ((lat_raw_std - lat_filtered_std) / lat_raw_std) * 100
    lon_std_improvement = ((lon_raw_std - lon_filtered_std) / lon_raw_std) * 100
    combined_std_improvement = ((combined_raw_std - combined_filtered_std) / combined_raw_std) * 100

    print(f"   {'Lat (degrees)':<15} {lat_raw_std:.8f}      {lat_filtered_std:.8f}      {lat_std_improvement:+.1f}%")
    print(f"   {'Lon (degrees)':<15} {lon_raw_std:.8f}      {lon_filtered_std:.8f}      {lon_std_improvement:+.1f}%")
    print(
        f"   {'Combined (deg)':<15} {combined_raw_std:.8f}      {combined_filtered_std:.8f}      {combined_std_improvement:+.1f}%")

    # Convert to meters for more intuitive understanding
    lat_raw_std_meters = lat_raw_std * 111000
    lon_raw_std_meters = lon_raw_std * 111000 * np.cos(np.deg2rad(df['latitude'].mean()))
    lat_filtered_std_meters = lat_filtered_std * 111000
    lon_filtered_std_meters = lon_filtered_std * 111000 * np.cos(np.deg2rad(df['latitude'].mean()))
    combined_raw_std_meters = np.sqrt(lat_raw_std_meters ** 2 + lon_raw_std_meters ** 2)
    combined_filtered_std_meters = np.sqrt(lat_filtered_std_meters ** 2 + lon_filtered_std_meters ** 2)

    print(f"\n   📏 IN METERS:")
    print(f"   {'Metric':<15} {'Raw':<15} {'Filtered':<15} {'Improvement':<15}")
    print(f"   {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15}")
    print(
        f"   {'Latitude':<15} {lat_raw_std_meters:.2f} m        {lat_filtered_std_meters:.2f} m        {lat_std_improvement:+.1f}%")
    print(
        f"   {'Longitude':<15} {lon_raw_std_meters:.2f} m        {lon_filtered_std_meters:.2f} m        {lon_std_improvement:+.1f}%")
    print(
        f"   {'Combined':<15} {combined_raw_std_meters:.2f} m        {combined_filtered_std_meters:.2f} m        {combined_std_improvement:+.1f}%")

    # 2. Noise Reduction Rate
    # Calculate variance of raw positions
    lat_raw_var = np.var(df['latitude'])
    lon_raw_var = np.var(df['longitude'])

    # Calculate variance of filtered positions
    lat_filtered_var = np.var(df['lat_kf'])
    lon_filtered_var = np.var(df['lon_kf'])

    # Calculate noise reduction percentages
    lat_noise_reduction = ((lat_raw_var - lat_filtered_var) / lat_raw_var) * 100
    lon_noise_reduction = ((lon_raw_var - lon_filtered_var) / lon_raw_var) * 100

    # Combined noise reduction (using combined variance)
    combined_raw_var = lat_raw_var + lon_raw_var
    combined_filtered_var = lat_filtered_var + lon_filtered_var
    combined_noise_reduction = ((combined_raw_var - combined_filtered_var) / combined_raw_var) * 100

    print(f"\n📊 NOISE REDUCTION RATE:")
    print(f"   Latitude:  {lat_noise_reduction:.2f}% reduction")
    print(f"   Longitude: {lon_noise_reduction:.2f}% reduction")
    print(f"   Combined:  {combined_noise_reduction:.2f}% reduction")

    # 3. Additional useful metrics
    print(f"\n📈 ADDITIONAL METRICS:")

    # Raw vs Filtered variance comparison
    print(f"   Raw Variance     - Lat: {lat_raw_var:.10f}, Lon: {lon_raw_var:.10f}")
    print(f"   Filtered Variance - Lat: {lat_filtered_var:.10f}, Lon: {lon_filtered_var:.10f}")

    # Position error (distance between consecutive points)
    def calculate_distances(lat, lon):
        """Calculate distances between consecutive GPS points in meters"""
        distances = []
        for i in range(1, len(lat)):
            # Haversine formula for distance calculation
            dlat = np.deg2rad(lat[i] - lat[i - 1])
            dlon = np.deg2rad(lon[i] - lon[i - 1])
            a = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.deg2rad(lat[i - 1])) * np.cos(np.deg2rad(lat[i])) *
                 np.sin(dlon / 2) ** 2)
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371000 * c  # Earth radius in meters
            distances.append(distance)
        return np.array(distances)

    raw_distances = calculate_distances(df['latitude'], df['longitude'])
    filtered_distances = calculate_distances(df['lat_kf'], df['lon_kf'])

    print(f"   Raw track smoothness (avg step): {np.mean(raw_distances):.2f} meters")
    print(f"   Filtered track smoothness (avg step): {np.mean(filtered_distances):.2f} meters")

    # Track length comparison
    print(f"   Raw track total length: {np.sum(raw_distances):.2f} meters")
    print(f"   Filtered track total length: {np.sum(filtered_distances):.2f} meters")

    # Data quality indicators
    print(f"\n📋 DATA QUALITY INDICATORS:")
    print(f"   Number of data points: {len(df)}")
    # print(f"   Time span: {df['fixtime'].iloc[-1] - df['fixtime'].iloc[0]}")
    print(f"   Average GPS accuracy: {df['accuracy'].mean():.2f} meters")
    print(f"   Average speed: {df['speed'].mean():.2f} m/s")

    print("\n" + "=" * 60)

    return {
        'lat_raw_std': lat_raw_std,
        'lon_raw_std': lon_raw_std,
        'lat_filtered_std': lat_filtered_std,
        'lon_filtered_std': lon_filtered_std,
        'combined_raw_std': combined_raw_std_meters,
        'combined_filtered_std': combined_filtered_std_meters,
        'lat_std_improvement': lat_std_improvement,
        'lon_std_improvement': lon_std_improvement,
        'combined_std_improvement': combined_std_improvement,
        'lat_noise_reduction': lat_noise_reduction,
        'lon_noise_reduction': lon_noise_reduction,
        'combined_noise_reduction': combined_noise_reduction
    }


if __name__ == '__main__':
    # Load deduplicated CSV
    df = pd.read_csv('newtrack_deduped.csv')

    # Apply the reordered CVA filter
    kf = KalmanFilterGPSCVAReorder(jerk_var=0.01)
    df['lat_kf'], df['lon_kf'] = kf.filter(df)

    calculate_metrics(df)

    plot_results(df)


