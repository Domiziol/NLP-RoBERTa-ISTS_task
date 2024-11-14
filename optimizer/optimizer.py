from yacs.config import CfgNode
from roberta_model import RobertaISTS
import torch


def build_optimizer(cfg: CfgNode, model: RobertaISTS.RobertaISTS) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
