import torch 
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import pa1_dataloader

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

# architecture
class Classifier(nn.Module):
    def __init__(self, hidden_1, hidden_2, hidden_3, hidden_4, dropout_prob):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(16*16, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, hidden_4)
        self.fc5 = nn.Linear(hidden_4, 10)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # flattening the input tensor
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        x = F.log_softmax(self.fc5(x), dim=1)
        
        return x

'''' MODEL, LOSS FUNCTION, AND OPTIMIZER '''

model = Classifier(hidden_1=512, hidden_2=512, hidden_3=256, hidden_4=128, dropout_prob=0.2)
print(model)

# =============================================================================
# if train_on_gpu:
#     model.cuda()
# =============================================================================

criterion = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.003)
optimizer = optim.SGD(model.parameters(), lr=0.003)


'''' TRAINING AND VALIDATION '''
# training
epochs = 100

# keeps minimum validation loss
valid_loss_min = np.Inf

train_losses, test_losses = [], []
for epoch in range(epochs):
    
    train_loss = 0
    
    # training the model
    
    model.train()
    for images, labels in train_loader:
# =============================================================================
#         if train_on_gpu:
#             images, labels = images.cuda(), labels.cuda()
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
               #         if train_on_gpu:
               #             images, labels = images.cuda(), labels.cuda()
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
            torch.save(model.state_dict(), 'mlp_model_USPS_SGD_mnist.pt')
            valid_loss_min = valid_loss
        model.train()
        
            
'''' TESTING '''
           
# testing
        
# testing on the test set 
# load last saved model with minimum valid_loss
model.load_state_dict(torch.load('mlp_model_USPS_SGD_mnist.pt'))

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
   for images, labels in test_loader:
       # =============================================================================
       # if train_on_gpu:
       #     images, labels = images.cuda(), labels.cuda()
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
import helper

images, labels = next(iter(test_loader))

# =============================================================================
# if train_on_gpu:
#     images, labels = images.cuda(), labels.cuda()
# =============================================================================

img = images[2].view(1, 256)

# turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)
    
# as output of the network is log probabity  we need to take the exp of the probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1,16,16), ps)       
        






        
        
        