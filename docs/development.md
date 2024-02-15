## Development

### Resources
- [Original GAN paper](https://arxiv.org/abs/1406.2661) by Goodfellow
- [Convolutions](https://docs.gimp.org/2.8/en/plug-in-convmatrix.html), [types](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) and[ more](- https://paperswithcode.com/method/convolution)
- [Affine transforms](https://in.mathworks.com/discovery/affine-transformation.html)
- [Divergence](https://maps.joindeltaacademy.com/?concept=58)

The GAN consists of two models -  a discriminator and a generator and by training both of these we want to make the output distribution of the generator match the distribution of the real images (reducing entropy). 

#### Discriminator
The discriminator used here uses convolutional layers to detect which distribution the input image is from. The input shape is (28,28,1) and the output is a single number between 0 and 1, denoting the probability that it is from the real dataset. It penalizes the generator for producing implausible results.

In each discriminator training step we take n samples from the real dataset and generate n fake samples, which are used by the discriminator to learn the correct weights. A very good discriminator means that the generator has to work harder to outwit it while a bad discriminator leads to the generator becoming lazy and generating some random output instead of what we desire (in this case it's digits).

#### Generator
The generator's job is to learn to generate plausible data to fool the discriminator. It takes as input a noise sample, in this example it is of the dimension (1,128), and learn a mapping of this by using transposed convolutions to output images which closely match the real distribution. 

### Training 
Since the input dataset (MNIST handwritten digits) has samples in the shape of (28,28,1), the output shape should be the same. The 1 at the end indicates one channel - a grayscale image.

GAN training proceeds in alternating periods:

1. The discriminator trains for one or more epochs.
2. The generator trains for one or more epochs.
3. Repeat steps 1 and 2 to continue to train the generator and discriminator networks.

During training of the generator the discriminator stays constant while the generator weight update is paused when training the discriminator. Since the goal of the GAN is to make the output distribution(generated) match the real image's distribution, we want the discriminator to be unable to distinguish between a generated and a real image when the model has converged.

This makes training GANs difficult as the generator is chasing a moving target if the discriminator gets too good, but if the generator gets too good the discriminator gives random feedback and training beyond this point reduces quality of generation.