
# coding: utf-8

# In[14]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


# In[23]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint 
        ## (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # The input to our layer is 112*112*1 as defined in the second notebook. We will feed this to the 
        # convolution kernel.In this case the convolution kernerl is composed of one channel for the gray scale
        # that we previously normalized, we also define a number of feature maps, in this case we use 16, a kernel
        # of size 5 that will serve as a filter, a stride of 1, which indicates how many pixels we will move along
        # and a padding of one pixel for when we hit the margin of the image. 
        self.conv1 = nn.Conv2d(in_channels=1,out_channels= 16, kernel_size=5, stride=1, padding=1)
        
        # The output of this convolution is determined by the kernel, stride and padding by the following formula.
        # O=(W-K+2P)/S+1; (112-5+2)/1+1=110. Since we have with and height plus a feature map the shape of the new
        # tensor to be fed to the ReLu is 34 (batch size), 110 (width), 110(heigh), 16 (feature map)/
        self.relu1=nn.ReLU()
        
        # After we feed the layer to the ReLU we get the same size as in the output of the convolution layer 34,110,110,16 
        # The maxpool layer requires we defined a kernel size and a stride. We can use the same formula as above to get
        # the output size of the height and the width. 
        self.maxpool1=nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ouput size 55,55,16
        
        # we finally feed output of the maxpool to the linear layer. We need to make sure to feed the three dimensions as
        # inputs. This layer will be squeezed to be fed into the linear layer using the view torch function (similar)
        # to the shape function of numpy. We use the functions view and size to get an array with 34 rows (or whatever 
        # number of samples we defined in the batch) and as many columns as in the previous step, in this case the input 
        # to our linear layer will be 55*55*16. We also defined how many units we want out.
        self.fc1=nn.Linear(55*55*16, 1024) 
        self.fc2=nn.Linear(1024, 512)
        self.fc3=nn.Linear(512,136)
        # We finally define a Dropout layer that will randomly get rid of values on our tensor. The value of 0.5 is a pretty
        # common standard.
        self.drop1=nn.Dropout(0.4)
        #self.drop2=nn.Dropout(0.5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

    
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #we define the forward behavior, where we feed the tensor through our different layers/
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.maxpool1(x)
        #print(x.shape)
        x = self.drop1(x)
        #print('shape of ReLu is', x.shape)
        #print(x.view(x.size(0),-1).shape)
        
        #squeeze the tensor so we can feed it to the linear activation.
        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        #x = self.drop2(x)
        x=self.fc2(x)
        x=self.fc3(x)       
        # a modified x, having gone through all the layers of your model, should be returned
        return x


# In[12]:


#?nn.Linear

