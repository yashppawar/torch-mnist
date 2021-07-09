from torch import nn, no_grad, float32

# TODO: make predict method
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        """Instantiate the Model"""
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        """
        run X from the layers to get prediction
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def compile(self, loss_fn, optimizer):
        """
        Set the loss function and optimizer for the model
        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def __train(self, dataloader, verbose=True):
        """
        Trains the neural network for 1 epoch, private method
        """
        size = len(dataloader.dataset)

        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction error
            pred = self(X)  # model(X)
            loss = self.loss_fn(pred, y)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                if verbose: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="\r")
        
        if verbose: print()

    def __test(self, dataloader, verbose=True):
        """
        tests the model on the given dataloader
        """
        size = len(dataloader.dataset)
        self.eval()  # model.eval()
        test_loss, correct = 0, 0

        with no_grad():  # dont calculate the gradients for validation set
            for X, y in dataloader:
                pred = self(X)  # model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(float32).sum().item()
        
        test_loss /= size
        correct /= size

        if verbose: print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def fit(self, train_dataloader, epochs=15, verbose=True, val_dataloader=None):
        """
        train the model on the given number of epochs
        """
        if not self.loss_fn or not self.optimizer:  # if model not compiled 
            print('\n\nModel not compiled\n\n')

        for epoch in range(epochs):  # train the model for given number of epochs
            print(f"\nEpoch {epoch+1}\n-------------------------------")
            self.__train(train_dataloader, verbose=verbose)

            if val_dataloader:  # run the validation if provided
                self.__test(val_dataloader, verbose=verbose)
