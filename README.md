# Aesthetic Adaption
### Implementation of a deep learning model that can adapt an existing work to resemble original aesthetic of any artist.

## Methodology
A deep learning model that can adapt an existing work to resemble the original aesthetic of any artist is called "Neural Style Transfer". NST is a computer vision technique that combines the content of one image with the style of another image, to create a new image that preserves the content of the original image while adopting the artistic style of a reference image. 

The technique involves training a deep neural network to separate the content and style features of images. The network consists of two parts: a "content" network that extracts the high-level features of an image, and a "style" network that captures the texture and color patterns of a style image. The two networks are then combined to generate a new image that has the content of the original image but with the style of the reference image. \
Refer this paper on [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)

## Implementation
Run the code as follows
```
python NeuralStyleTransfer.py content.jpg style.jpg
```
where ```content.jpg``` and ```style.jpg``` refer to the original reference image and artistic style image respectively. You can test with any content and style image of your choice by changing the argument path in the command line

Note : Hyperparameters ```alpha``` and ```beta``` control the amount of style and content in the target image. 
* Increasing the ratio ```alpha/beta``` increases the retention of content structure in the target image
* Decreasing the ratio ```alpha/beta``` increases the retention of style layout in the target image

## Results
![generated-stack-1.png](/)
![generated-stack-2.png](/)

## Alternate Solutions
There are several variations of Neural Style Transfer, but one of the most common approaches is to use a generative adversarial network (GAN), specifically a conditional GAN (cGAN), that conditions on the original image and generates a new image that has the desired artistic style. The GAN is trained with a loss function that encourages the generated image to match the style of the reference image while preserving the content of the original image.

