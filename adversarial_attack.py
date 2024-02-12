""""
The task involves developing a program that manipulates images by adding adversarial noise. This noise is designed to trick an image classification model into misclassifying the altered image as a specified target class, regardless of the original content.
You may select any pre-trained image classification model for this task. A model from the torchvision library is recommended, but not mandatory.
The core challenge is to effectively introduce noise into the image in such a way that the model misclassifies it as the desired target class.

Input:
The user will provide an image and specify a target class.

Output:
The program should output an image that has been altered with adversarial noise. The altered image should be classified by the model as the target class, irrespective of the original image's content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

EPSILON_START = 0.01
EPSILON_INCREMENT = 0.01
MAX_ATTEMPTS = 10
orignal_model_path = "models/lenet_mnist_model.pth"

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def perform_attack(image, target_class):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is: {device}.")
    model = Model().to(device)
    model.load_state_dict(torch.load(orignal_model_path, map_location=device))
    model.eval()

    attempts_cnt = 0
    success = False
    epsilon = EPSILON_START
    last_adversarial_image = ()
    while attempts_cnt < MAX_ATTEMPTS and not success:

        image, target_class = image.to(device), target_class.to(device)
        image.requires_grad = True
        output = model(image)
        prediction = output.max(1, keepdim=True)[1]

        # If the image already clasisfied as target label, no need for attack
        if prediction.item() == target_class.item():
            success = True
            return image
        else:
            loss = F.nll_loss(output, target_class)
            model.zero_grad()
            loss.backward()
            gradient = image.grad.data

            print(f"Attempt number: {attempts_cnt}. Epsilon: {epsilon}")
            adversarial_image = get_adversarial_image(image, epsilon, gradient)
            last_adversarial_image = (attempts_cnt, adversarial_image.squeeze().detach().cpu().numpy())
            output = model(adversarial_image)
            prediction = output.max(1, keepdim=True)[1]

            if prediction.item() == target_class.item():
                success = True
                return adversarial_image
            else:
                attempts_cnt += 1
                epsilon += EPSILON_INCREMENT
                print(f"Failure: Output label is: {prediction}")

    print("Attack unsuccessful.")

    attempt_no, image = last_adversarial_image
    plt.ylabel(f"Attempt no: {attempt_no}")
    plt.imshow(image, cmap="gray")
    plt.show()


def get_adversarial_image(image, epsilon, gradient):
    gradient_sign = gradient.sign()
    adversarial_image = image + epsilon * gradient_sign
    return adversarial_image


if __name__ == '__main__':
    # Get an image from MNIST dataset
    mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

    image, label =  next(iter(mnist_loader))
    print(f"Gold label class is: {label}.")
    target_label = '2'
    target = torch.tensor([int(target_label)])
    print(target)
    adversarial_image = perform_attack(image, target)

 
