import torch 
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import pa1_dataloader
from torch.nn.modules.utils import _pair
# =============================================================================
# '''' CHECKING IF GPU IS AVAILABLE '''
# 
# check if CUDA is availableå
train_on_gpu = torch.cuda.is_available()
 
if train_on_gpu:
    print('CUDA is available. Training on GPU...')
else:
    print('CUDA is not available. Training on CPU...')
# =============================================================================

'''' DATASETS AND DATALOADERS '''

BATCH_SIZE = 16
# training data
train_loader = pa1_dataloader.load(train = True, batch_size=BATCH_SIZE)

# testing data
test_loader = pa1_dataloader.load(train = False, batch_size=BATCH_SIZE)

'''' VISUALIZING DATA '''

# visualize data

# batch view
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25,4))
for i in np.arange(16):
    ax = fig.add_subplot(2,16/2, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[i]), cmap='gray')
    ax.set_title(str(labels[i].item()))
    
# detailed view 
img = np.squeeze(images[1]) 
fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height =img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val= round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    color = 'white' if img[x][y]<thresh else 'black')
        
'''' NETWORK ARCHITECTURE '''
# (((W-F+2*P )/S)+1) 
# W = width of image, F = kernel dimension, P = padding, S = stride
# architecture
class LocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocalConv2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class CNN_Local_USPS(nn.Module):
    def __init__(self, fc_1_output, dropout_prob):
        super(CNN_Local_USPS, self).__init__()
        
        self.conv1 = LocalConv2d(1, 8, 14, 3, 1)
        self.conv2 = LocalConv2d(8, 16, 5, 3, 1)
        self.conv3 = LocalConv2d(16, 32, 3, 3, 1)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(32*3*3, fc_1_output)
        self.fc2 = nn.Linear(fc_1_output, 10)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flattening the input tensor
        x = x.view(-1, 32*3*3)
        
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

'''' MODEL, LOSS FUNCTION, AND OPTIMIZER '''

model = CNN_Local_USPS(fc_1_output=256, dropout_prob=0.2)
print(model)

# =============================================================================
if train_on_gpu:
    model.cuda()
# =============================================================================

criterion =nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.003)
optimizer = optim.SGD(model.parameters(), lr=0.01)


'''' TRAINING AND VALIDATION '''
# training
epochs = 50

# keeps minimum validation loss
valid_loss_min = np.Inf

train_losses, test_losses = [], []
for epoch in range(epochs):
    
    train_loss = 0
    
    # training the model
    
    model.train()
    for images, labels in train_loader:
# =============================================================================
        if train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
# =============================================================================
            
        optimizer.zero_grad()
        
        output = model(images)   
        loss = criterion(output, labels.long())
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
    
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
           for images, labels in test_loader:
               # =============================================================================
               if train_on_gpu:
                   images, labels = images.cuda(), labels.cuda()
               # =============================================================================
               output = model(images)
               valid_loss += criterion(output, labels.long())
               
               ps = torch.exp(output)
               top_p, top_class = ps.topk(1, dim=1)
               equals = top_class == labels.view(*top_class.shape)
               accuracy += torch.mean(equals.type(torch.FloatTensor))
       
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(valid_loss/len(test_loader))
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(test_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving Model ...'.format(
                    valid_loss_min,
                    valid_loss))
            torch.save(model.state_dict(), 'cnn__local_model_USPS_SGD_mnist.pt')
            valid_loss_min = valid_loss
        model.train()
        
            
'''' TESTING '''
           
# testing
        
# testing on the test set 
# load last saved model with minimum valid_loss
model.load_state_dict(torch.load('cnn__local_model_USPS_SGD_mnist.pt'))

model.eval()
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
   for images, labels in test_loader:
       # =============================================================================
       if train_on_gpu:
           images, labels = images.cuda(), labels.cuda()
       # =============================================================================
       output = model(images)
       test_loss += criterion(output, labels.long())
       
       ps = torch.exp(output)
       top_p, top_class = ps.topk(1, dim=1)
       equals = top_class == labels.view(*top_class.shape)
       accuracy += torch.mean(equals.type(torch.FloatTensor))

print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

# inference

images, labels = next(iter(test_loader))

# =============================================================================
if train_on_gpu:
    images, labels = images.cuda(), labels.cuda()
# =============================================================================

# turn off gradients to speed up this part
with torch.no_grad():
    logps = model(images)
    
# as output of the network is log probabity  we need to take the exp of the probabilities
ps = torch.exp(logps)
top_p, top_class = ps.topk(1, dim=1)
equals = top_class == labels.view(*top_class.shape)

images, labels = images.cpu(), labels.cpu()

top_class = top_class.cpu()
top_class =top_class.numpy().tolist()

equals = equals.cpu()
equals = equals.numpy().tolist()

# idx = 0 -> (BATCH_SIZE-1)
idx = 10
if idx < BATCH_SIZE:
    plt.imshow(images[idx].view(16,16))
    plt.title('Predicted: ' + str(top_class[idx][0]) + ' -> ' + str(equals[idx][0]))
else:
    print('Output of BATCH_SIZE index')
        
        






        
        
        