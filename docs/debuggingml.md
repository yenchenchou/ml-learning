# Debugging Nerual Networks

Let's think about the components in neural networks from a perspective of debugging neural networks. I will try to list out all things I can think of to debug the model.

1. Debugging model input
    - We can use the model to train on smaller data to see if it overfit. We can condlude that the model is learn if it coverfits.

2. Weight initialization
    - Take a look at the weight after with several epochs. We can look at the weight to see of the weight is roughly normal distributed. Also, check the gradients to see if gradient vanishing or explosion happens along with several epoches of observation.

3. Debugging the loss
    - We can observe the loss function plot across epoches on both training data and validation data. A good quality training is when the training loss and validation loss line are roughly overlap; if the validation loss line is always a bit lower then it's a bit underfitting, and this can be solved by higher learning rate, change model activation function, and choose bigger parameters; if the validation loss is always higher it is a bit overfitting, and we can use drop out, normalization, regularization; if the validation loss drop then climb up then it's overfitting, we can lower or use learning rate decay to address the issue.
