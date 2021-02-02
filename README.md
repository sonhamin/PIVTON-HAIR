# PIVTON-HAIR : Pose Invariant Virtual Try-on Hair (with Conditional Image Completion)

### Inspired by [PITVONS](https://link.springer.com/chapter/10.1007/978-3-030-20876-9_41), we created a method to view faces with different hairstyles

![Alt text](images/flow.jpg?raw=true)

<p> Given two inputs: a segmented face and segmented hair, a generator outputs an image that captures the features of the both inputs, allowing one to view a face with a different hairstyle. <br/>
It is near impossible to perform supervised learning on a problem such as this - as it is difficult to obtain a dataset that can suffice.
Thus we adapt a triple-loss / image completion learning method for swapping hair. 
  
The training method extremely similar to PIVTONS.</p>

1. We set up our problem by segmenting the faces and hair from the CelebA dataset. Here, faces are segmented down to the eyebrows fromt he top, and up to the ears at the sides. This is to decrease the possibility of bangs being included in the face segmentation.
2. Three models are created
    1. Generator - Reconstructs an image of a full face given a segmented face and segmented hair in an attempt to fool the discriminator
    2. Discriminator - Binary classifier trying to distinguish original image from fake (generator output) ones
    3. Pre-trained model - Model used to calculate perceptual loss by comparing feature map of generator
3. A training step consists of 
    1. Calculating the adversarial loss -- how well disciminator is fooled by generated image
    2. Calculating the perceptual loss -- difference of feature maps of real/generated image are
    3. Calculating the l2-loss -- difference of pixels between real/generated
    4. Updating the generator based on all three losses

![Alt text](images/training.png?raw=true)
#### Image from [PITVONS](https://link.springer.com/chapter/10.1007/978-3-030-20876-9_41)

#### ** The model is far from perfect - especially with biases towards white female celebrities. A more diverse dataset towards the target audience would be the most optimal solution
