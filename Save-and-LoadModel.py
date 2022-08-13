'''
本节参考官方文档：https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
'''

import torch
import torchvision.models as models

# Saving and Loading Model Weights
# save
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), "model_weights.pth")
# load
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Saving and Loading Models with Shapes
# save
torch.save(model, 'model.pth')
# load
model = torch.load('model.pth')