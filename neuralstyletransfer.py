# Import the necessary libraries
import torch.nn as nn
from torchvision import models, transforms, utils
from PIL import Image
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('content_path', type=str)
parser.add_argument('style_path', type=str)
args = parser.parse_args()

# Pretrained VGG19 model with selected layers
class VGG19(nn.Module):
  def __init__(self):
    super(VGG19, self).__init__()

    # [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
    self.select_layers = [0, 5, 10, 19, 28]
    
    # Retrieve only feature layers till conv5_1
    self.model = models.vgg19(pretrained=True).features[:29]

  def forward(self, x):
    # list to store the features for layers : [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
    feats = []
    for layer_idx, layer in enumerate(self.model):
      x = layer(x)

      if layer_idx in self.select_layers:
        feats.append(x) 

    return feats

# Define Content Loss per layer
def content_loss(gen_feature, content_feature):

    # MSE Loss between the individual feature maps of the layer
    loss = torch.mean((gen_feature - content_feature)**2)
    return loss

# Define Style per layer features
def style_loss(gen_feature, style_feature):

    # Unroll the features
    b, c, h, w = gen_feature.shape
    gen_feature = gen_feature.view(c, h*w)

    b, c, h, w = style_feature.shape
    style_feature = style_feature.view(c, h*w)

    # Compute Gram Matrix 
    A = torch.mm(gen_feature, gen_feature.t())
    G = torch.mm(style_feature, style_feature.t())

    # MSE Loss between gram matrix of generated features and original style features
    loss = torch.mean((A - G)**2)
    return loss

def total_loss(gen_features, content_features, style_features, alpha, beta):
    loss_c = loss_s = 0

    # Aggregate the content and style loss across all the layers
    for gen_feature, content_feature, style_feature in zip(gen_features, content_features, style_features):
        loss_c += content_loss(gen_feature, content_feature)
        loss_s += style_loss(gen_feature, style_feature) 

    loss_t = alpha*loss_c + beta*loss_s
    return loss_t

# Define the training parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10000
lr = 1e-3
alpha = 1
beta = 100

# Load the content and style image pair
def load_image_as_tensor(imagepath, transform, device):
    img = Image.open(imagepath)
    img_t = transform(img)
    return img_t.unsqueeze(0).to(device=device)

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
])

# Define the input content and style image paths
content_path = args.content_path
style_path = args.style_path
content_img = load_image_as_tensor(content_path, transform, device)
style_img = load_image_as_tensor(style_path, transform, device)

# Define the generated image to be trained as the content image
generated_img = content_img.clone().requires_grad_(True)

# Initialize the weight freezed VGG feature model and optimizer
model = VGG19().to(device=device).eval()
optimizer = torch.optim.Adam([generated_img], lr=lr)

# Compute the content and style feature layers before training
content_feats = model(content_img)
style_feats = model(style_img)

# Train the reverse network i.e we freeze the network weights and train the pixel values of input image
for epoch in range(epochs):

    # Extract the features
    generated_feats = model(generated_img)

    # Calculate Loss
    train_loss = total_loss(generated_feats, content_feats, style_feats, alpha, beta)

    # Optimize
    optimizer.zero_grad()
    train_loss.backward(retain_graph=True)
    optimizer.step()

    if epoch%100==0:
        print(f'Epoch : {epoch}, loss : {train_loss}')
        utils.save_image(generated_img, 'generated.png')

utils.save_image(torch.cat([content_img, style_img, generated_img]), 'generated_stack.png')

