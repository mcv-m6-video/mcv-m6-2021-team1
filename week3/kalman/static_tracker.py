


class StaticTracker:

    def __init__(self):
        self.bbox = None

    def init(self, frame, bbox):
        self.bbox = bbox

    def update(self, frame):
        return True, self.bbox