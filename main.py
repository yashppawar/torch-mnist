import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from data import load_train_data, load_test_data
from model import NeuralNetwork

# define the constants
BATCH_SIZE = 32
EPOCHS = 15

# load the data
train_dataloader = load_train_data(BATCH_SIZE, shuffle=True)
test_dataloader = load_test_data(BATCH_SIZE)

# instantiate the model
model = NeuralNetwork()

print('************************************************\n\n\n')

loss_fn = CrossEntropyLoss()
learning_rate = 1e-3
optimiser = SGD(model.parameters(), lr=learning_rate)

model.compile(loss_fn, optimiser)

print(model)

model.fit(
    train_dataloader,
    epochs=EPOCHS,
    val_dataloader=test_dataloader
)

print("Done!")
