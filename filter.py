import math

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.s = None

    def filter(self, x):
        if self.s is None:
            self.s = x
        else:
            self.s = self.alpha * x + (1 - self.alpha) * self.s
        return self.s

    def set_alpha(self, alpha):
        self.alpha = alpha


class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(self.alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self.alpha(d_cutoff))
        self.last_x = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.last_x is None:
            dx = 0.0
        else:
            dx = (x - self.last_x) * self.freq
        self.last_x = x

        edx = self.dx_filter.filter(dx)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        self.x_filter.set_alpha(self.alpha(cutoff))
        return self.x_filter.filter(x)


class OneEuroFilter2D:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.fx = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)

    def filter(self, u, v):
        return self.fx.filter(u), self.fy.filter(v)


class LandmarkFilterManager:
    def __init__(self, num_points=21, freq=30, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.filters = [
            OneEuroFilter2D(freq, min_cutoff, beta, d_cutoff)
            for _ in range(num_points)
        ]

    def filter(self, landmarks):
        """
        landmarks: [(u, v, z), ...]  # 원본 입력
        returns:  [(u_filtered, v_filtered, z_raw), ...]
        """
        return [
            (*self.filters[i].filter(u, v), z)
            for i, (u, v, z) in enumerate(landmarks)
        ]
