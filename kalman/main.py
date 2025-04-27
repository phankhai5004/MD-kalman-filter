import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and trim dataset
df = pd.read_csv("newtrack.csv").head(1000)

# Parse time
df['fixtime'] = pd.to_datetime(df['fixtime'])

# Extract variables
lat = np.array(df["latitude"])
lon = np.array(df["longitude"])
speed = np.array(df["speed"])
course = np.array(df["course"])

# Convert speed and course to vx (latspeed) and vy (longspeed)
vx = speed * np.cos(np.deg2rad(course))
vy = speed * np.sin(np.deg2rad(course))

# Observations: [lat, lon]
observations = np.stack((lat, lon), axis=1)

# Initial state: [lat, lon, vx, vy]
initial_state = np.array([lat[0], lon[0], vx[0], vy[0]])

# Compute delta times (in seconds)
dt_array = df['fixtime'].diff().dt.total_seconds().fillna(1.0).values

print(dt_array)


class KalmanFilter:
    def __init__(self, initial_state):
        self.ndim = 4

        # Initial state
        self.mu_hat = initial_state

        # Initial covariance matrix
        self.P = np.eye(4)

        # Measurement noise covariance (R)
        self.R = np.eye(2) * 1e-4

        # Observation matrix (H)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

    def predict(self, dt):
        # Update transition matrix F with dynamic dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Process noise covariance (Q)
        process_noise_pos = 0.1
        process_noise_vel = 0.2
        Q = np.array([
            [dt ** 4 / 4 * process_noise_pos, 0, dt ** 3 / 2 * process_noise_pos, 0],
            [0, dt ** 4 / 4 * process_noise_pos, 0, dt ** 3 / 2 * process_noise_pos],
            [dt ** 3 / 2 * process_noise_pos, 0, dt ** 2 * process_noise_vel, 0],
            [0, dt ** 3 / 2 * process_noise_pos, 0, dt ** 2 * process_noise_vel]
        ])

        self.mu_hat = F @ self.mu_hat
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        y = z - self.H @ self.mu_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.mu_hat = self.mu_hat + K @ y
        self.P = (np.eye(self.ndim) - K @ self.H) @ self.P
        return self.mu_hat.copy()


# Initialize Kalman filter
kf = KalmanFilter(initial_state)

# Apply Kalman filter
predictions = []
for i, z in enumerate(observations):
    dt = dt_array[i] if i > 0 else 1.0  # use 1.0 for the first step
    kf.predict(dt)
    pred = kf.update(z)
    predictions.append(pred)

# Convert predictions to DataFrame
pred_df = pd.DataFrame(predictions, columns=["latitude", "longitude", "latspeed", "longspeed"])
obs_df = pd.DataFrame(observations, columns=["latitude", "longitude"])

# Save results
pred_df.to_csv("Prediction.csv", index=False)
obs_df.to_csv("Observation.csv", index=False)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(obs_df["latitude"], obs_df["longitude"], label="Observed", linestyle="-", color="red")
print(obs_df["latitude"])
print(pred_df["latitude"])

plt.plot(pred_df["latitude"], pred_df["longitude"], label="Kalman Prediction", linestyle="-", color="blue")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Kalman Filter with Variable Time Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# print("jalkdsjflkdsajfldsa")
