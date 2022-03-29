from load_crowd_counting_model import get_model
from torchvision import transforms

# Access commons
model = get_model()
# Standard RGB transform
transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize(mean=[0.302234709263, 0.291243076324, 0.269087553024],
                                                       std=[0.227743327618, 0.211051672697, 0.184846073389]),
                                 ])

def get_image_prediction(image):
    img = transform(image)
    img = img.cpu()
    # img = img[None,:,:,:].cuda()
    output = model(img.unsqueeze(0))
    # output = model(img)
    prediction = int(output.detach().cpu().sum().numpy()/100) 
    density_map = output.detach().cpu().numpy()[0][0]
    return prediction, density_map