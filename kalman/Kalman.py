import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("newtrack.csv")


df = df.head(1000)


lat = np.array([df.latitude])


long = np.array([df.longitude])

vel = np.array([df.speed])

course = np.array([df.course])

lng = len(lat[0])


coord1 = [
    list(i)
    for i in zip(lat[0], long[0], vel[0] * np.cos(course[0]), vel[0] * np.sin(course[0]))
]

Raw_data = pd.DataFrame(coord1, columns=["latitude", "longitude", "latspeed", "longspeed"])
Raw_data.to_csv("Raw.csv")

coord = list(zip(lat[0], long[0], vel[0], vel[0]))

from pylab import *
from numpy import *
import matplotlib.pyplot as plt


class Kalman:
    def __init__(self, ndim):
        self.ndim = ndim
        self.Sigma_x = eye(ndim) * 1e-4  # Process noise (Q)
        self.F = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]  # Transition matrix which predict state for next time step (F)
        self.H = [[1, 0, 0, 0], [0, 1, 0, 0]]  # Observation matrix (H)
        self.mu_hat = 0  # State vector (X)
        self.cov = eye(ndim)  # Process Covariance (P)
        self.R = 1e-4  # Sensor noise covariance matrix / measurement error (R)

    def update(self, obs):

        # Make prediction
        # x = F.x
        # P = F.P.Ft + Q

        self.mu_hat_est = dot(self.F, self.mu_hat)
        self.cov_est = dot(self.F, dot(self.cov, transpose(self.F))) + self.Sigma_x

        # Update estimate
        # y = z - H.x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        self.error_mu = obs - dot(self.H, self.mu_hat_est)
        self.error_cov = dot(dot(self.H, self.cov), transpose(self.H)) + self.R
        self.K = dot(dot(self.cov_est, transpose(self.H)), linalg.inv(self.error_cov))
        self.mu_hat = self.mu_hat_est + dot(self.K, self.error_mu)
        if ndim > 1:
            self.cov = dot((eye(self.ndim) - dot(self.K, self.H)), self.cov_est)
        else:
            self.cov = (1 - self.K) * self.cov_est


coord_output = []
cov = []

for coordinate in coord1:
    temp_list = []
    cov_list = []
    ndim = 4
    nsteps = 100
    k = Kalman(ndim)
    mu_init = np.array(coordinate)  # initial data
    cov_init = 0.0001 * ones((ndim))  # initial covariance
    obs = zeros((ndim, nsteps))  # initiate a matrix observation
    for t in range(nsteps):
        obs[:, t] = random.normal(
            mu_init, cov_init
        )  # Generate observation by normal distribution
    for t in range(ndim, nsteps):
        k.update(obs[:, t])  # Starting an update and also make decision
        # print ("Actual: ", obs[:, t], "Prediction: ", k.mu_hat_est[0])
    # k.update(mu_init)

    temp_list.append(obs[:, t])
    temp_list.append(k.mu_hat_est[0])
    # cov_list.append(k.cov_est)
    cov.append(k.cov_est)
    coord_output.append(temp_list)


df2 = pd.DataFrame(coord_output)
df2.to_csv("coord_output.csv")

# df3= pd.DataFrame(cov)
# df3.to_csv('cov_output.csv')

Actual = df2[0]  # contain noise
Prediction = df2[1]  # contain prediction
# Covariance = df3[0]

Actual_df = pd.DataFrame(Actual)
Prediction_df = pd.DataFrame(Prediction)
# Covariance_df = pd.DataFrame(Covariance)

# Raw_data = pd.DataFrame(coord1, columns = ['latitude', 'longitude','speed'])

Actual_coord = pd.DataFrame(
    Actual_df[0].to_list(), columns=["latitude", "longitude", "latspeed", "longspeed"]
)
Actual_coord.to_csv("Actual_noise.csv")

Prediction_coord = pd.DataFrame(
    Prediction_df[1].to_list(),
    columns=["latitude", "longitude", "latspeed", "longspeed"],
)
Prediction_coord.to_csv("Prediction.csv")


plt.plot(
    Actual_coord["latitude"],
    Actual_coord["longitude"],
    label="Random Observation",
    linestyle=":",
)
plt.plot(
    Prediction_coord["latitude"],
    Prediction_coord["longitude"],
    label="Prediction",
    linestyle="-",
)
plt.plot(Raw_data["latitude"], Raw_data["longitude"], label="Raw", linestyle="--")

plt.xlabel("latitude")
plt.ylabel("longitude")
plt.title("Comparison of Raw and Prediction (nstep = 100)")
plt.legend()
plt.show()
# ----------------------------

plt.plot(
    Prediction_coord["latitude"],
    Prediction_coord["longitude"],
    label="Prediction",
    linestyle="-",
)

plt.xlabel("latitude")
plt.ylabel("longitude")
plt.title("Prediction")
plt.legend()
plt.show()

plt.plot(
    Raw_data["latitude"], Raw_data["longitude"], label="Raw", linestyle="-", color="red"
)
plt.xlabel("latitude")
plt.ylabel("longitude")
plt.title("Raw Data")
plt.legend()
plt.show()

# plt.plot(Actual_coord ['speed'], label='Noise',linestyle=':')
# plt.plot(Prediction_coord['speed'], label='Prediction', linestyle='-')
# plt.plot(Prediction_df[1].to_list()[2] + 2*np.sqrt(cov[2,2]), label='Prediction', linestyle='--')
# plt.plot(Prediction_coord['speed'] + 2*np.sqrt(cov[2,2]), label='Prediction', linestyle='--')
# plt.plot(Raw_data['latspeed'], label='latspeed', linestyle='--')
# # plt.plot(Raw_data['longspeed'], label='longspeed', linestyle='--')
# plt.title('Speed')
# plt.legend()
# plt.show()

# # plt.plot(Raw_data['latspeed'], label='latspeed', linestyle='--')
# plt.plot(Raw_data['longspeed'], label='longspeed', linestyle='--',color='red')
# plt.title('Speed')
# plt.legend()
# plt.show()
