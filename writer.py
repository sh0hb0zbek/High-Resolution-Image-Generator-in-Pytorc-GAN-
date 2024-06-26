from tensorboardX import SummaryWriter


def writer(path):
    return SummaryWriter(log_dir=path)