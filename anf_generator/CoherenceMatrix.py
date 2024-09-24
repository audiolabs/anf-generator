import numpy as np
from scipy.special import jv


class Parameters:
    def __init__(self, mic_positions, sc_type, sample_frequency, nfft=1024, c=340, speed=20, direction=60):
        self.mic_positions = mic_positions  # Microphone positions [M x 3]
        # Spatial coherence model: 'corcos', 'spherical', or 'cylindrical'
        self.sc_type = sc_type.lower()
        self.sample_frequency = sample_frequency  # Sample frequency (Hz)
        self.nfft = nfft  # FFT length
        self.c = c  # Sound velocity (m/s)
        self.speed = speed  # Wind speed (m/s)
        self.direction = direction  # Wind direction (degrees)

    def copy(self):
        return Parameters(
            mic_positions=self.mic_positions.copy(),
            sc_type=self.sc_type,
            sample_frequency=self.sample_frequency,
            nfft=self.nfft,
            c=self.c,
            speed=self.speed,
            direction=self.direction
        )


class CoherenceMatrix:
    def __init__(self, params: Parameters):
        self.params = params
        self.matrix = self.generate_target_coherence()

    def generate_target_coherence(self):
        """
        Generate the analytical target coherence based on the model
        (e.g., diffuse spherical/cylindrical, Corcos), the sensor positions, 
        and the FFT length. Valid for an arbitrary 3D-array geometry.

        Returns:
            DC (np.array): Target coherence matrix [Channels x Channels x nfft/2+1]
        """

        # Angular frequency vector
        ww = 2 * np.pi * self.params.sample_frequency * \
            np.arange(self.params.nfft // 2 + 1) / self.params.nfft

        # Matrix of position vectors
        rr = self.params.mic_positions[:, None, :] - \
            self.params.mic_positions[None, :, :]

        # Matrix of inter-sensor distances
        d = np.linalg.norm(rr, axis=2)

        if self.params.sc_type == 'corcos':
            # Desired wind speed and direction
            Ud = self.params.speed / 3.6  # Wind speed [m/s]
            theta = np.deg2rad(self.params.direction)  # Angle in radians

            # Rotation matrix for wind direction
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
            yy = np.array([0, 1, 0])  # y axis = true North
            u = np.dot(yy, R)  # Wind direction unit vector
            # Wind direction unit perpendicular vector
            u_p = np.array([u[1], -u[0], 0])

            # Coherence parameters
            alpha1, alpha2 = -0.125, -0.7
            alpha__ = alpha1 * np.abs(np.sum(u[None, None, :] * rr, axis=2)) \
                + alpha2 * np.abs(np.sum(u_p[None, None, :] * rr, axis=2))
            im__ = np.sum(u[None, None, :] * rr, axis=2)
            U = 0.8 * Ud  # Convective turbulence speed
            AA = np.einsum('ij,k->ijk', alpha__ - 1j * im__, ww)
            DC = np.exp(AA / U)

        elif self.params.sc_type == 'spherical':
            DC = np.sinc(d[:, :, None] * ww[None, None, :] /
                         (self.params.c * np.pi))

        elif self.params.sc_type == 'cylindrical':
            DC = jv(0,d[:, :, None] * ww[None, None, :] /
                         (self.params.c))

        else:
            raise ValueError(
                'Unknown coherence model. Please select "corcos", "spherical", or "cylindrical".')

        return DC
