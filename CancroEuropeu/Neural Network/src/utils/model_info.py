from torch.utils.tensorboard import SummaryWriter
class Info():
    Writer = SummaryWriter()
    Epoch = 0
    Device = ''
    Name = ''
    Optim = ''
    WeightDecay = 0
    LR_Decay = 0
    LR = 0
    Momentum = 0
    SaveType = ''

    BoardX = f"{Name}/SavedWith_{SaveType}/{Optim}/LR_{LR}/Momentum_{Momentum}"
    PATH = f'D:/output/{Name}/SaveType_{SaveType}/{Optim}/lr_{LR}/momentum_{Momentum}'