from torch.utils.tensorboard import SummaryWriter

class Info():
    Writer = SummaryWriter()
    Model = ''
    Optim = ''
    WeightDecay = 0
    LR_Decay = 0
    LR = 0
    Momentum = 0
    SaveType = ''