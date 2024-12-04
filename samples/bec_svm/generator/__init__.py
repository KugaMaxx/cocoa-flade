from . import generator_ops


class BinaryEventContour(generator_ops.Detector):
    def __init__(self, 
                 resolution, 
                 threshold: float = 0.88, 
                 candidate_num: int = 1, 
                 min_area: int = 25) -> None:
        super().__init__(resolution, 
                         threshold=threshold, 
                         candidate_num=candidate_num,
                         min_area=min_area)

    def detect(self, sample):
        if sample['events'] is None:
            return []

        return super().detect(sample['events'])
