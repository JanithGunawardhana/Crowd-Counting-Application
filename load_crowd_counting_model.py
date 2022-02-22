import torch

from crowd_counting_model import TestNet

def get_model():
    model = TestNet()
    model.load_state_dict(torch.load('./crowd_counting_model.pth', map_location='cpu')) # Where we upload our model (Download model to local)
    model.eval()
    return model