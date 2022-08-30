from models.loss import model_psmnet_loss, stereo_psmnet_loss
from models.loss import model_gwcnet_loss
from models.MyNet import MyNet

__models__ = {
    "mynet": MyNet
}

__loss__ = {
    "mynet": stereo_psmnet_loss
}