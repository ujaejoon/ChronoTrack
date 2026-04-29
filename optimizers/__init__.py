import torch


def AdamW(cfg, params):
    return torch.optim.AdamW(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)


def Adam(cfg, params):
    return torch.optim.Adam(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)


def SGD(cfg, params):
    return torch.optim.SGD(params=params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)


def create_optimizer(cfg, params):
    type2optimizer = dict(Adam=Adam, SGD=SGD, AdamW=AdamW)
    return type2optimizer[cfg.optimizer_type](cfg, params)
