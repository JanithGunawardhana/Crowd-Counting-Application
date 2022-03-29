import torch

from crowd_counting_model import TestNet

def get_model():
    model = TestNet()
    #model = TestNet().cuda()
    model.load_state_dict(torch.load('./crowd_counting_model.pth', map_location='cpu'))
    # model.load_state_dict(torch.load('./crowd_counting_model.pth')) 
    model.eval()
    return model