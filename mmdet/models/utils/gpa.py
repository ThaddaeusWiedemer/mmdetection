from mmcv.runner import BaseModule


class GPAHead(BaseModule):
    def __init__(self, init_cfg=None):
        super(GPAHead, self).__init__(init_cfg)