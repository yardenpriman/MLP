
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

data_path = "./MNIST_data"

# Defining a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the data
mnist_data = datasets.MNIST(data_path, download=True, train=True, transform=transform)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)

# get single batch
dataiter = iter(mnist_dataloader)
batch_images, batch_labels = next(dataiter)

# Print the number of samples in the whole dataset
num_samples = len(mnist_dataloader.dataset)
print("number of samples in the whole dataset:", num_samples)

# Print the number of samples in a single batch
num_samples_in_batch = len(batch_images)
print("number of samples in a single batch:", num_samples_in_batch)

# Print the shape of images in the data (image dimensions)
image_shape = batch_images[0].shape
print("shape of images in the data (image dimensions):", image_shape)

# Print the number of labels in the whole dataset
num_labels = len(mnist_dataloader.dataset.targets)
print("number of labels in the whole dataset:", num_labels)

# plot three images and print their labels
idx = np.random.choice(range(64),3) # three rundom indices
plt.subplot(1,3,1)
plt.imshow(batch_images[idx[0]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1,3,2)
plt.imshow(batch_images[idx[1]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1,3,3)
plt.imshow(batch_images[idx[2]].numpy().squeeze(), cmap='Greys_r')
print("Labels:",batch_labels[idx])

#  Feed-Forward Neural Network - Architecture

from torch import nn, optim
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        '''
        Declare layers for the model
        '''
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
          nn.Linear(784, 128),  # Input layer to first hidden layer
          nn.ReLU(),  # ReLU activation for first hidden layer
          nn.Linear(128, 64),  # First hidden layer to second hidden layer
          nn.ReLU(),  # ReLU activation for second hidden layer
          nn.Linear(64, 10),  # Second hidden layer to output layer
          nn.LogSoftmax(dim=1)  # Log-softmax activation for output layer
        )
        #self.log_softmax = nn.LogSoftmax(dim=1)  # Log-softmax activation for output layer


    def forward(self, x):
        ''' Forward pass through the network, returns log_softmax values '''
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    image - the input image to the network
    ps - the class confidences (network output)
    '''
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


def random_prediction_example(data_loader, model):
  '''
  The function sample an image from the data, pass it through the model (inference)
  and show the prediction visually. It returns the predictions confidences.
  '''
  # take a batch and randomly pick an image
  dataiter = iter(data_loader)
  images, labels = next(dataiter)
  images.resize_(64, 1, 784)
  img = images[0]

  # Forward pass through the network
  # I use torch.no_grad() for faster inference and to avoid gradients from moving through the network.
  with torch.no_grad():
      ps = model(img)
      # the network outputs log-probabilities, so take exponential for probabilities
      ps = torch.exp(ps)

  # visualize image and prediction
  view_classify(img.view(1, 28, 28), ps)
  return ps

preds_conf = random_prediction_example(mnist_dataloader, model)


# Printing the prediction of the network for that sample:
print(preds_conf)
print(int(torch.argmax(preds_conf)))

# Feed-Forward Neural Network - Training

from torch.utils import data

# split trainset into train and validation
train_set, validation_set = torch.utils.data.random_split(mnist_data, [0.8, 0.2])

# create data loader for the trainset
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# create data loader for the valset
val_loader = DataLoader(validation_set, batch_size=64, shuffle=False)

# set hyper parameters
learning_rate = 0.003
nepochs = 5

model = NeuralNetwork()

# create sgd optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# create a criterion object
criterion = nn.NLLLoss()

# Train the model
def train_model(model, optimizer, criterion,
                nepochs, train_loader, val_loader, is_image_input = False):
  '''
  Train a pytorch model and evaluate it every epoch.
  Params:
  model - a pytorch model to train
  optimizer - an optimizer
  criterion - the criterion (loss function)
  nepochs - number of training epochs
  train_loader - dataloader for the trainset
  val_loader - dataloader for the valset
  is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                Should be True for Neural Networks
                                but False for Convolutional Neural Networks.
  '''
  train_losses, val_losses = [], []
  for e in range(nepochs):
      running_loss = 0
      running_val_loss = 0
      for images, labels in train_loader:
          if is_image_input:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

          # Training pass
          model.train() # set model in train mode
          optimizer.zero_grad()  # Zero the parameter gradients
          # forward pass
          log_ps = model(images)
          # calculate loss
          loss = criterion(log_ps, labels)
          # backward pass
          loss.backward()
          # update weights
          optimizer.step()

          running_loss += loss.item()
      else:
          val_loss = 0
          # Evalaute model on validation at the end of each epoch.
          with torch.no_grad():
              for images, labels in val_loader:
                  if is_image_input:
                    # Flatten MNIST images into a 784 long vector
                    images = images.view(images.shape[0], -1)
                  model.eval()  # Set model to evaluation mode
                  log_ps = model(images)
                  val_loss = criterion(log_ps, labels)
                  running_val_loss += val_loss.item()

          # track train loss and validation loss
          train_losses.append(running_loss/len(train_loader))
          val_losses.append(running_val_loss/len(val_loader))

          print("Epoch: {}/{}.. ".format(e+1, nepochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                "Validation Loss: {:.3f}.. ".format(running_val_loss/len(val_loader)))
  return train_losses, val_losses

# Training the model
train_losses, val_losses = train_model(model, optimizer, criterion, nepochs,
                                       train_loader, val_loader, is_image_input=True)

# plot train and validation loss as a function of #epochs
# Plotting the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, nepochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, nepochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch [Number]')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# if you run this line multiple times you will get different images
random_prediction_example(mnist_dataloader, model)

# Calculate the model's accuracy on the validation-set

def evaluate_model(model, val_loader, is_image_input=False):
  '''
  Evaluate a model on the given dataloader.
  Params:
  model - a pytorch model to train
  val_loader - dataloader for the valset
  is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                   Should be True for Neural Networks
                                   but False for Convolutional Neural Networks.
  '''
  validation_accuracy = 0
  with torch.no_grad():
      for images, labels in val_loader:
          if is_image_input:
            # flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
          # forward pass
          log_ps = model(images)
          ps = torch.exp(log_ps)
          top_p, top_class = ps.topk(1, dim=1)
          # count correct predictions
          equals = top_class == labels.view(*top_class.shape)

          validation_accuracy += torch.sum(equals.type(torch.FloatTensor))
  res = validation_accuracy/len(val_loader.dataset)
  return res

print(f"Validation accuracy: {evaluate_model(model, val_loader, is_image_input=True)}")

# Convolutional Neural Networks
## Prepocess
# Defining the Convolutional Neural Network
class ConvolutionalNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
new_cnn_model = ConvolutionalNet()
print(new_cnn_model)

# set hyperparameters
cnn_nepochs = 5
cnn_learning_rate = 0.01

# train the conv model
new_cnn_model = ConvolutionalNet()
# create sgd optimizer
cnn_optimizer = optim.SGD(new_cnn_model.parameters(), lr=cnn_learning_rate)
# create negative log likelihood loos
cnn_criterion = nn.NLLLoss()

train_losses, val_losses = train_model(new_cnn_model, cnn_optimizer, cnn_criterion,
                                       cnn_nepochs, train_loader, val_loader, is_image_input=False)

# Save the best model
best_model = new_cnn_model
