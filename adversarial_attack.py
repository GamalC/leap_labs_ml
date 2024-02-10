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
import torch.nn.functional as F

EPSILON_START = 0.1
EPSILON_INCREMENT = 0.01
MAX_ATTEMPTS = 10
orignal_model_path = "models/lenet_mnist_model.pth"

def perform_attack(image, target_class):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model.load_state_dict(torch.load(orignal_model_path, map_location=device))
    model.eval()

    attempts_cnt = 0
    success = False
    epsilon = EPSILON_START
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

            print("Attempt number: {attempts_cnt}. Epsilon: {epsilon}")
            adversarial_image = get_adversarial_image(image, epsilon, gradient)
            output = model(adversarial_image)
            prediction = output.max(1, keepdim=True)[1]

            if prediction.item() == target_class.item():
                success = True
                return adversarial_image
            else:
                attempts_cnt += 1
                epsilon += EPSILON_INCREMENT


def get_adversarial_image(image, epsilon, gradient):
    print("do attack")
    adversarial_image = image
    return adversarial_image
 
