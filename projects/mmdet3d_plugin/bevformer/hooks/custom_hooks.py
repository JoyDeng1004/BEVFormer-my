from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            # runner.model.state_dict()--obtain the weights of the current model
            # runner.eval_model.load_state_dict()--load these weights into the evaluation model
            runner.eval_model.load_state_dict(runner.model.state_dict())
