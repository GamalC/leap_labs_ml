""""
The task involves developing a program that manipulates images by adding adversarial noise. This noise is designed to trick an image classification model into misclassifying the altered image as a specified target class, regardless of the original content.
You may select any pre-trained image classification model for this task. A model from the torchvision library is recommended, but not mandatory.
The core challenge is to effectively introduce noise into the image in such a way that the model misclassifies it as the desired target class.

Input:
The user will provide an image and specify a target class.

Output:
The program should output an image that has been altered with adversarial noise. The altered image should be classified by the model as the target class, irrespective of the original image's content.
"""