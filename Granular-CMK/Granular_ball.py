import numpy as np


class GrainBall:
    def __init__(self, center, radius, weight):
        self.center = np.array(center)  # 使用 numpy 处理多维坐标
        self.radius = radius
        self.weight = weight
        if self.radius == 0:
            self.density = 0
        else:
            self.density = self.weight / self.radius

    def __repr__(self):
        return f"GrainBall(center={self.center}, radius={self.radius}, weight={self.weight})"
