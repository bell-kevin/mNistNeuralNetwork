<a name="readme-top"></a>

# 

Neural network trained on the MNIST dataset for digit classification (0–9). It is a feedforward neural network (multi-layer perceptron) with one hidden layer of size 128 and uses ReLU in the hidden layer, softmax in the output layer, and cross-entropy loss. We implement it from scratch using only NumPy (no high-level frameworks like TensorFlow/PyTorch).

Note: Training a neural network on the full MNIST dataset purely in Python/NumPy without any optimizations is quite slow. This example code is primarily for educational purposes—it shows how backpropagation and gradient descent can be implemented manually. If you run it, consider using fewer samples or fewer epochs to speed things up.

Key Points

1. Data:

   - We load the MNIST dataset (60,000 training, 10,000 test images).
     
   - Each image is 28×28 pixels, flattened into a 784-dimensional vector.
     
   - Labels are integers from 0 to 9 (we one-hot encode them for training).

  2. Network Architecture:
     
     - Input Layer: 784 units (one per pixel).
    
     - Hidden Layer: 128 units, ReLU activation.
       
     - Output Layer: 10 units (one per digit class), softmax activation to get probability distribution.

3. Loss: Cross-entropy is used for multi-class classification.

4. Training:
   - We use mini-batch gradient descent.
     
   - For each mini-batch, we do a forward pass, compute loss, do a backward pass (using chain rule), and then update weights with a simple gradient descent step.

5. Performance:
   
   - Typically, you can achieve 90%+ accuracy with even a modest 1-hidden-layer MLP. Properly tuned and trained, you can surpass 95%.
    
   - This pure NumPy version is not optimized, so it will be slow on large datasets.

6. Improvements (if you want to go further):
   
   - Add more hidden layers.
   
   - Use advanced optimizers (e.g. Adam, RMSProp).
     
   - Add batch normalization or dropout.
  
   - Experiment with learning rate schedules.
   
--------------------------------------------------------------------------------------------------------------------------
== We're Using GitHub Under Protest ==

This project is currently hosted on GitHub.  This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Souce Software
(FOSS).  We are deeply concerned about using a proprietary system like GitHub
to develop our FOSS project. I have a [website](https://bellKevin.me) where the
project contributors are actively discussing how we can move away from GitHub
in the long term.  We urge you to read about the [Give up GitHub](https://GiveUpGitHub.org) campaign 
from [the Software Freedom Conservancy](https://sfconservancy.org) to understand some of the reasons why GitHub is not 
a good place to host FOSS projects.

If you are a contributor who personally has already quit using GitHub, please
email me at **bellKevin@pm.me** for how to send us contributions without
using GitHub directly.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission.  We do not consent to GitHub's use of this project's
code in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
