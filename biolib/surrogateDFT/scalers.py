import json


class MinMaxScaler:
    def __init__(self):

        self.property = "e_gap_min"
        self.min = 0.004
        self.max = 0.183

    def __str__(self):
        return f"MinMaxScaler(property={self.property}, min={self.min}, max={self.max})"

    def scale(self, sample: float):
        return 2 * (sample - self.min) / (self.max - self.min) - 1

    def unscale(self, sample: float):
        return ((sample + 1) * (self.max - self.min) / 2) + self.min
