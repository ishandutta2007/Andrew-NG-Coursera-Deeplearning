# 👁️ Visual Recognition with Convolutional Networks

Course four of the deep learning specialization offered on [Coursera](https://www.coursera.org/specializations/deep-learning), presented by [DeepLearning.ai](http://deeplearning.ai/). Lectures delivered by Andrew Ng.

# 📚 Deep Learning Guide Collection
* [**Foundations of Neural Nets**](guides/01-foundations-of-neural-nets.md)
* [**Tuning & Optimization**](guides/02-tuning-and-optimization.md)
* [**Project Strategy Playbook**](guides/03-project-strategy-playbook.md)
* [**Visual Recognition Networks**](guides/04-visual-recognition-networks.md)
* [**Temporal Sequence Learning**](guides/05-temporal-sequence-learning.md)

## 📑 Table of Contents

* [Visual Recognition with Convolutional Networks](#-visual-recognition-with-convolutional-networks)
   * [Table of Contents](#-table-of-contents)
   * [What This Course Covers](#-what-this-course-covers)
   * [🏗️ Building Blocks of CNNs](#️-building-blocks-of-cnns)
      * [🖼️ Understanding Computer Vision](#️-understanding-computer-vision)
      * [🔲 Detecting Edges in Images](#-detecting-edges-in-images)
      * [🔳 Adding Padding to Inputs](#-adding-padding-to-inputs)
      * [🦶 Using Strided Convolutions](#-using-strided-convolutions)
      * [🧊 Working with Volumetric Convolutions](#-working-with-volumetric-convolutions)
      * [🧱 Anatomy of a Single Conv Layer](#-anatomy-of-a-single-conv-layer)
      * [📐 Walkthrough: A Basic ConvNet](#-walkthrough-a-basic-convnet)
      * [🏊 Downsampling with Pooling](#-downsampling-with-pooling)
      * [🔬 Full CNN Architecture Example](#-full-cnn-architecture-example)
      * [💡 Advantages of Convolutional Layers](#-advantages-of-convolutional-layers)
   * [🏛️ Landmark CNN Architectures](#️-landmark-cnn-architectures)
      * [📖 Why Study Existing Architectures?](#-why-study-existing-architectures)
      * [🏆 Foundational Network Designs](#-foundational-network-designs)
      * [🔗 Skip Connections & ResNets](#-skip-connections--resnets)
      * [🤔 Why Residual Networks Succeed](#-why-residual-networks-succeed)
      * [🔬 1×1 Convolutions & Network-in-Network](#-11-convolutions--network-in-network)
      * [💡 The Inception Module Concept](#-the-inception-module-concept)
      * [🌐 The Inception Network (GoogLeNet)](#-the-inception-network-googlenet)
      * [📂 Leveraging Open-Source Code](#-leveraging-open-source-code)
      * [🔄 Transfer Learning Strategies](#-transfer-learning-strategies)
      * [🖼️ Expanding Data with Augmentation](#️-expanding-data-with-augmentation)
      * [📊 Current Landscape of Computer Vision](#-current-landscape-of-computer-vision)
   * [🎯 Detecting Objects in Images](#-detecting-objects-in-images)
      * [📍 Locating Objects in Images](#-locating-objects-in-images)
      * [📌 Detecting Key Landmarks](#-detecting-key-landmarks)
      * [🔍 Sliding Window Object Detection](#-sliding-window-object-detection)
      * [⚡ Conv-Based Sliding Windows](#-conv-based-sliding-windows)
      * [📦 Predicting Bounding Boxes](#-predicting-bounding-boxes)
      * [📏 Intersection Over Union (IoU)](#-intersection-over-union-iou)
      * [🚫 Non-Maximum Suppression](#-non-maximum-suppression)
      * [⚓ Using Anchor Boxes](#-using-anchor-boxes)
      * [🎯 The Complete YOLO Pipeline](#-the-complete-yolo-pipeline)
      * [🔲 Region-Based CNNs (R-CNN Family)](#-region-based-cnns-r-cnn-family)
   * [🎭 Advanced Applications: Faces & Artistic Style](#-advanced-applications-faces--artistic-style)
      * [👤 Recognizing Faces](#-recognizing-faces)
         * [🤷 Face Recognition Explained](#-face-recognition-explained)
         * [☝️ Learning from a Single Example](#️-learning-from-a-single-example)
         * [🔀 The Siamese Architecture](#-the-siamese-architecture)
         * [🔺 Triplet Loss Function](#-triplet-loss-function)
         * [✅ Binary Classification for Face Verification](#-binary-classification-for-face-verification)
      * [🎨 Artistic Neural Style Transfer](#-artistic-neural-style-transfer)
         * [🖌️ What is Neural Style Transfer?](#️-what-is-neural-style-transfer)
         * [🧠 Visualizing What ConvNets Learn](#-visualizing-what-convnets-learn)
         * [💰 The Overall Cost Function](#-the-overall-cost-function)
         * [📝 Content Similarity Cost](#-content-similarity-cost)
         * [🎭 Style Similarity Cost](#-style-similarity-cost)
         * [📐 Extending Convolutions to 1D and 3D](#-extending-convolutions-to-1d-and-3d)
   * [🧩 Additional Topics](#-additional-topics)
      * [🐍 Working with Keras](#-working-with-keras)

## 🎬 What This Course Covers

Below is the official course description from the course [page](https://www.coursera.org/learn/convolutional-neural-networks):

> This course will teach you how to build convolutional neural networks and apply it to image data. Thanks to deep learning, computer vision is working far better than just two years ago, and this is enabling numerous exciting applications ranging from safe autonomous driving, to accurate face recognition, to automatic reading of radiology https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/. 
>
> You will:
> - Understand how to build a convolutional neural network, including recent variations such as residual networks.
> - Know how to apply convolutional networks to visual detection and recognition tasks.
> - Know to use neural style transfer to generate art.
> - Be able to apply these algorithms to a variety of image, video, and other 2D or 3D data.
>
> This is the fourth course of the Deep Learning Specialization.



## 🏗️ Building Blocks of CNNs

> Master the core layers of CNNs — including convolutions and pooling — and learn to assemble them into deep architectures for multi-class image classification.

### 🖼️ Understanding Computer Vision

- Computer vision stands out as one of the fastest-growing fields powered by deep learning.
- Key real-world applications leveraging deep learning for vision include:
  - Autonomous vehicles.
  - Facial recognition systems.
- Deep learning has also unlocked new creative possibilities in digital art.
- The rapid progress in computer vision continues to unlock applications that were unimaginable just a few years back.
- Innovations from computer vision frequently transfer to other domains and inspire novel architectural ideas.
  - As an example, Andrew Ng adapted certain computer vision concepts for use in speech recognition.
- Common categories of computer vision tasks:
  - Image classification.
  - Object detection.
    - Identifying objects and pinpointing their positions.
  - Neural style transfer.
    - Applying the artistic style of one image onto another.
- A major challenge in computer vision is that https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ can be extremely large, and we need algorithms that are both quick and precise.
  - For instance, a `1000x1000` image yields 3 million input features for a fully connected network. If the next hidden layer has 1000 neurons, the weight matrix would be `[1000, 3 million]` — that's 3 billion parameters in just the first layer, which is computationally prohibitive!
- The answer is to construct networks using **convolutional layers** rather than **fully connected layers**.

### 🔲 Detecting Edges in Images

- The convolution operation is a core building block of any CNN. A classic illustration is edge detection in images.
- In the early layers, a CNN typically learns edges; in the middle layers, it identifies object parts; and in the deeper layers, it assembles these parts into complete representations.
- Within an image, we can identify vertical edges, horizontal edges, or edges at any orientation.
- Vertical edge detection:
  - An illustration of the convolution operation for finding vertical edges:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//01.png)
  - In this case, a `6x6` matrix convolved with a `3x3` filter/kernel produces a `4x4` output.
  - In TensorFlow, the relevant function is `tf.nn.conv2d`. In Keras, you'd use `Conv2d`.
  - The vertical edge filter identifies `3x3` regions where a bright area transitions to a dark area.
  - When this filter slides over a light-to-dark boundary, positive values emerge. Over a dark-to-light boundary, negative values appear. Using the absolute value function resolves the sign ambiguity.
- Horizontal edge detection
  - The filter looks like this:

    ```
    1	1	1
    0	0	0
    -1	-1	-1
    ```

- Numerous filter configurations exist for detecting horizontal or vertical edges. Here's the **Sobel** vertical filter (which gives extra weight to the center row):

  ```
  1	0	-1
  2	0	-2
  1	0	-1
  ```

- There's also the **Scharr** filter (which emphasizes the center row even more):

  ```
  3	0	-3
  10	0	-10
  3	0	-3
  ```

- A key insight from deep learning: rather than hand-designing these filter values, we can treat them as learnable parameters. The network can automatically discover horizontal, vertical, diagonal, or any other edge pattern on its own.

### 🔳 Adding Padding to Inputs

- Padding is essential for building deep neural networks effectively.
- Recall from the previous section: a `6x6` input convolved with a `3x3` filter yields a `4x4` result.
- The general rule: an `nxn` matrix convolved with an `fxf` kernel produces an `n-f+1, n-f+1` output. 
- The convolution operation reduces spatial dimensions whenever f>1.
- When applying convolution repeatedly, the image shrinks at each step, discarding valuable information. Additionally, edge pixels participate in fewer convolution operations than central pixels.
- Two drawbacks of basic convolutions:
  - Output dimensions shrink.
  - Edge information gets underutilized.
- The fix is to pad the input image before convolving by appending extra rows and columns. The padding amount `P` indicates how many rows/columns are added to the top, bottom, left, and right.
- Padding values are almost always zeros.
- Updated general rule: an `nxn` matrix convolved with an `fxf` kernel using padding `p` produces an `n+2p-f+1, n+2p-f+1` output. 
- With n = 6, f = 3, and p = 1 the result is `n+2p-f+1 = 6+2-3+1 = 6`. The spatial dimensions are preserved.
- "Same" convolution means padding is chosen so that output dimensions match input dimensions. The formula is:

  ```
  P = (f-1) / 2
  ```

- In computer vision, f is typically odd — one reason being that an odd-sized filter has a well-defined center pixel.

### 🦶 Using Strided Convolutions

- Strided convolution is another fundamental component found in CNNs.

- The stride value `S` determines how many pixels the filter jumps at each step during convolution. In our previous examples, S was 1.

- The updated general formula:
  
-  An `nxn` input convolved with an `fxf` filter, padding `p`, and stride `s` yields an output of size `(n+2p-f)/s + 1, (n+2p-f)/s + 1`. 
  
- When `(n+2p-f)/s + 1` is not a whole number, we apply the **floor** function.

- In mathematics textbooks, the convolution operation involves flipping the filter before applying it. What we do in deep learning is technically cross-correlation, but the community refers to it simply as convolution.

- For "same" convolutions (output equals input size), the padding formula becomes:

  ```
  p = (n*s - n + f - s) / 2
  When s = 1 ==> P = (f-1) / 2
  ```

### 🧊 Working with Volumetric Convolutions

- We've covered how convolution operates on 2D https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/; now let's explore convolution on 3D https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ (such as RGB images).
- We convolve an image with dimensions (height, width, # channels) using a filter with matching (height, width, same # channels). The channel count of the filter must equal the channel count of the input.
- Think of this as having stacked filters, one for each channel!
- Example:
  - Input image: `6x6x3`
  - Filter: `3x3x3`
  - Result image: `4x4x1`
  - In this result p=0, s=1
- Note that the output is 2D in this case.
- By employing multiple filters, we can detect various features simultaneously. Example:
  - Input image: `6x6x3`
  - 10 Filters: `3x3x3`
  - Result image: `4x4x10`
  - In this result p=0, s=1

### 🧱 Anatomy of a Single Conv Layer

- The process: convolve a set of filters with the input, add a bias term to each filter's output, then apply the RELU activation. Example:
  - Input image: `6x6x3`         `# a0`
  - 10 Filters: `3x3x3`         `#W1`
  - Result image: `4x4x10`     `#W1a0`
  - Add b (bias) with `10x1` will get us : `4x4x10` image      `#W1a0 + b`
  - Apply RELU will get us: `4x4x10` image                `#A1 = RELU(W1a0 + b)`
  - In this result p=0, s=1
  - Note: total parameters here are `(3x3x3x10) + 10 = 280`
- This constitutes one layer in a CNN.
- Important observation: regardless of input dimensions, the parameter count stays constant for a given filter size. This property helps prevent overfitting.
- Standard notation for a convolutional layer l:

  ```
  Hyperparameters
  f[l] = filter size
  p[l] = padding	# Default is zero
  s[l] = stride
  nc[l] = number of filters

  Input:  n[l-1] x n[l-1] x nc[l-1]	Or	 nH[l-1] x nW[l-1] x nc[l-1]
  Output: n[l] x n[l] x nc[l]	Or	 nH[l] x nW[l] x nc[l]
  Where n[l] = (n[l-1] + 2p[l] - f[l] / s[l]) + 1

  Each filter is: f[l] x f[l] x nc[l-1]

  Activations: a[l] is nH[l] x nW[l] x nc[l]
  		     A[l] is m x nH[l] x nW[l] x nc[l]   # In batch or minbatch training
  		     
  Weights: f[l] * f[l] * nc[l-1] * nc[l]
  bias:  (1, 1, 1, nc[l])
  ```

### 📐 Walkthrough: A Basic ConvNet

- Let's construct a larger example step by step.
  - Input Image:   `a0 = 39x39x3`
    - `n0 = 39` and `nc0 = 3`
  - First layer (Conv layer):
    - `f1 = 3`, `s1 = 1`, and `p1 = 0`
    - `number of filters = 10`
    - Output: `a1 = 37x37x10`
      - `n1 = 37` and `nc1 = 10`
  - Second layer (Conv layer):
    - `f2 = 5`, `s2 = 2`, `p2 = 0`
    - `number of filters = 20`
    - Output: `a2 = 17x17x20`
      - `n2 = 17`, `nc2 = 20`
    - Note: dimensions shrink much faster with stride 2
  - Third layer (Conv layer):
    - `f3 = 5`, `s3 = 2`, `p3 = 0`
    - `number of filters = 40`
    - Output: `a3 = 7x7x40`
      - `n3 = 7`, `nc3 = 40`
  - Fourth layer (Fully connected Softmax):
    - `a3 = 7x7x40 = 1960`  flattened to a vector.
- Notice how spatial dimensions progressively decrease across layers — this is a common pattern.
- Categories of layers in a convolutional network:
  - Convolution. 		`#Conv`
  - Pooling      `#Pool`
  - Fully connected     `#FC`

### 🏊 Downsampling with Pooling

- Beyond convolution layers, CNNs frequently employ pooling layers to shrink spatial dimensions, accelerate computation, and make detected features more robust.
- Max pooling illustration:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//02.png)
  - This example uses `f = 2`, `s = 2`, and `p = 0` as hyperparameters.
- The intuition behind max pooling: if a feature is detected anywhere within a given region, retain its strongest activation. The practical reason for using pooling is that it consistently works well and reduces computation.
- Max pooling has zero learnable parameters.
- Max pooling on a 3D input:
  - Input: `4x4x10`
  - `Max pooling size = 2` and `stride = 2`
  - Output: `2x2x10`
- Average pooling computes the mean of values rather than taking the maximum.
- Max pooling is preferred over average pooling in most practical scenarios.
- When the pooling stride matches the filter size, the spatial dimensions are halved.
- Hyperparameter recap:
  - f : filter size.
  - s : stride.
  - Padding is seldom used in pooling.
  - Choice between max or average pooling.

### 🔬 Full CNN Architecture Example

- Let's walk through a complete CNN architecture resembling ***LeNet-5***, designed by Yann LeCun.
  - Input Image:   `a0 = 32x32x3`
    - `n0 = 32` and `nc0 = 3`
  - First layer (Conv layer):        `#Conv1`
    - `f1 = 5`, `s1 = 1`, and `p1 = 0`
    - `number of filters = 6`
    - Output: `a1 = 28x28x6`
      - `n1 = 28` and `nc1 = 6`
    - Then apply (Max pooling):         `#Pool1`
      - `f1p = 2`, and `s1p = 2`
      - Output: `a1 = 14x14x6`
  - Second layer (Conv layer):   `#Conv2`
    - `f2 = 5`, `s2 = 1`, `p2 = 0`
    - `number of filters = 16`
    - Output: `a2 = 10x10x16`
      - `n2 = 10`, `nc2 = 16`
    - Then apply (Max pooling):         `#Pool2`
      - `f2p = 2`, and `s2p = 2`
      - Output: `a2 = 5x5x16`
  - Third layer (Fully connected)   `#FC3`
    - Number of neurons: 120
    - Output `a3 = 120 x 1` . 400 came from `5x5x16`
  - Fourth layer (Fully connected)  `#FC4`
    - Number of neurons: 84
    - Output `a4 = 84 x 1` .
  - Fifth layer (Softmax)
    - Number of neurons is 10 (e.g., for classifying the 10 digits).
- Convention: Conv1 + Pool1 together count as a single layer.
- Statistics for this architecture:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//03.png)
- There are many hyperparameters to set. Refer to guidelines discussed later or draw inspiration from published literature.
- A general trend: spatial dimensions shrink as you go deeper, while filter count increases.
- Typically a CNN stacks one or more convolution operations followed by a pooling step.
- Fully connected layers tend to hold the majority of a network's parameters.
- When deciding how to combine these building blocks, study existing successful architectures first for intuition.

### 💡 Advantages of Convolutional Layers

- Two key benefits of convolutions:
  - Parameter sharing.
    - A feature detector (like a vertical edge detector) that proves useful in one region of an image is likely useful in other regions too.
  - Sparsity of connections.
    - Each output value in a layer depends on only a small subset of inputs, which promotes translation invariance.
- The complete picture:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//04.png)

## 🏛️ Landmark CNN Architectures

> Explore the practical techniques and insights used in deep CNN designs, drawn directly from influential research papers.

### 📖 Why Study Existing Architectures?

- We've covered convolutional, pooling, and fully connected layers individually. Computer vision researchers have invested years figuring out how to combine them effectively.
- Studying real-world examples is the best way to build intuition.
- Architectures that excel at one task often transfer well to others.
- Several foundational CNN networks worth studying:
  - **LeNet-5**
  - **AlexNet**
  - **VGG**
- The ImageNet competition winner **ResNet** featured an impressive 152 layers!
- Google's **Inception** architecture is another highly instructive design to learn and apply.
- Reading and experimenting with these models will sharpen your skills and spark ideas for your own projects.

### 🏆 Foundational Network Designs

- This section covers the classic architectures: **LeNet-5**, **AlexNet**, and **VGG**.

- **LeNet-5**

  - Designed to recognize handwritten digits from `32x32x1` grayscale images. Here's a diagram:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//05.png)
  - Published in 1998. The original didn't use softmax in the output layer.
  - Contains approximately 60k parameters.
  - Image dimensions decrease while channel counts grow across layers.
  - `Conv ==> Pool ==> Conv ==> Pool ==> FC ==> FC ==> softmax` — this arrangement is very standard.
  - The original paper employed Sigmoid and Tanh activations. Today's implementations predominantly use RELU.
  - [[LeCun et al., 1998. Gradient-based learning applied to document recognition]](http://ieeexplore.ieee.org/document/726791/?reload=true)

- **AlexNet**

  - Named after Alex Krizhevsky, the lead author. Geoffrey Hinton was among the co-authors.

  - Built to tackle the ImageNet challenge — classifying https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ into 1000 categories. Architecture diagram:

  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//06.png)

  - Overview:

    - ```
      Conv => Max-pool => Conv => Max-pool => Conv => Conv => Conv => Max-pool ==> Flatten ==> FC ==> FC ==> Softmax
      ```

  - Structurally similar to LeNet-5, but substantially larger.

  - Contains 60 million parameters (compared to LeNet-5's 60k).

  - Adopted the RELU activation function.

  - The original paper utilized multiple GPUs and Local Response Normalization (LRN).

    - Multiple GPUs were necessary because hardware was slower at the time.
    - Later studies showed that Local Response Normalization provides minimal benefit, so it can be safely ignored.

  - This paper was pivotal in persuading the computer vision community of deep learning's importance.

  - [[Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

- **VGG-16**

  - An evolution of AlexNet focused on architectural simplicity.
  - Rather than many different hyperparameters, it uses a consistent set of building blocks:
    - CONV = 3 X 3 filter, s = 1, same  
    - MAX-POOL = 2 X 2 , s = 2
  - Architecture diagram:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//07.png)
  - A large network even by today's standards with roughly 138 million parameters.
    - The bulk of these reside in fully connected layers.
  - Requires around 96MB of memory per image during forward propagation alone!
    - Most of the memory consumption occurs in the earlier layers.
  - Filter count doubles progressively: 64 → 128 → 256 → 512. The 512-filter stage is repeated twice.
  - Pooling alone handles spatial downsampling.
  - **VGG-19** is a deeper variant, but VGG-16 achieves comparable results and is more widely used.
  - The VGG paper established helpful design principles for CNN construction.
  - [[Simonyan & Zisserman 2015. Very deep convolutional networks for large-scale image recognition]](https://arxiv.org/abs/1409.1556)

### 🔗 Skip Connections & ResNets

- Extremely deep networks are hard to train due to vanishing and exploding gradient issues.
- This section introduces skip connections, which allow activations from one layer to bypass intermediate layers and feed directly into a much deeper layer — enabling the training of networks with 100+ layers.
- **Residual block**
  - ResNets are composed of these residual blocks.
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//08.png)
  - A shortcut/skip connection is inserted before the second activation function.
  - The inventors demonstrated that stacking these blocks enables training of significantly deeper networks.
  - [[He et al., 2015. Deep residual networks for image recognition]](https://arxiv.org/abs/1512.03385)
- **Residual Network**
  - A neural network built by stacking multiple residual blocks.
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//09.png)
  - These networks can grow deeper without performance degradation. In standard (plain) networks, theory predicts better solutions with more depth, but vanishing/exploding gradients undermine actual performance. Residual connections solve this and allow networks of arbitrary depth.
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//10.png)
  - Left: plain network; Right: ResNet. Clearly, the ResNet maintains or improves performance as depth increases.
  - In certain problems, additional depth may not yield further improvement — it depends on the specific task.
  - Some researchers have experimented with 1000-layer networks, though these aren't common in production.
  - [He et al., 2015. Deep residual networks for image recognition]

### 🤔 Why Residual Networks Succeed

- Let's walk through an example illustrating why ResNets are effective.

  - Start with a large network:

    - `X --> Big NN --> a[l]`

  - Now append two layers as a residual block:

    - `X --> Big NN --> a[l] --> Layer1 --> Layer2 --> a[l+2]`
    - With a direct connection from a`[l]` to `a[l+2]`

  - Assume we use RELU activations.

  - Then:

    - ```
      a[l+2] = g( z[l+2] + a[l] )
      	   = g( W[l+2] a[l+1] + b[l+2] + a[l] )
      ```

  - If L2 regularization drives `W[l+2]` toward zero and `b[l+2]` is also zero:

  - Then `a[l+2] = g( a[l] ) = a[l]` (since there are no negative values with RELU).

  - This demonstrates that learning the identity function is trivial for a residual block — which is why deeper networks remain trainable.

  - The two appended layers never degrade the original network's performance.

  - Important: z[l+2] and a[l] must share the same dimensions. When they differ, a learnable (or fixed) parameter matrix is introduced:

    - `a[l+2] = g( z[l+2] + ws * a[l] ) # ws adjusts dimensions to match`
    - ws can also be implemented via zero padding.

- Skip connections facilitate gradient flow during backpropagation, which is the fundamental reason deeper networks can be trained.

- Let's examine ResNet applied to https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/.

  - The **ResNet-34** architecture:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//resNet.jpg)
  - All 3x3 Convs use "same" padding.
  - Design philosophy: keep things simple.
  - Rule of thumb: when spatial dimensions halve, double the filter count.
  - No FC layers, no dropout.
  - Two primary block types are used depending on whether input/output dimensions match. You'll implement both.
  - Dotted lines indicate dimension mismatches. These are resolved by downsampling the input by 2 and zero-padding. An alternative approach called bottleneck is explored later.

- Helpful concept (**Spectrum of Depth**):

  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//12.png)
  - Source: [icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf](icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf)

- Residual block variants:

  - Identity block:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//16.png)
    - Note: each conv is followed by batch normalization (`BN`) then `RELU`. Dimensions remain unchanged.
    - This skip bridges 2 layers. Skip connections can span n layers where n>2.
    - This diagram follows [Keras](https://keras.io/) layer conventions.
  - The convolutional block:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//17.png)
    - The conv can be a bottleneck 1 x 1 conv

### 🔬 1×1 Convolutions & Network-in-Network

- A 1 x 1 convolution — also known as Network in Network — plays a crucial role in many modern CNN designs.

- What exactly does a 1 x 1 convolution accomplish? Isn't it just scalar multiplication?

  - Consider this first example:
    - Input: `6x6x1`
    - Conv: `1x1x1` one filter.        `# The 1 x 1 Conv`
    - Output: `6x6x1`
  - Now a multi-channel example:
    - Input: `6x6x32`
    - Conv: `1x1x32` 5 filters.     `# The 1 x 1 Conv`
    - Output: `6x6x5`

- Originally proposed in [Lin et al., 2013. Network in network].

- Widely adopted in modern architectures like ResNet and Inception.

- A 1 x 1 convolution is valuable when:

  - You need to reduce the channel dimension. This is also called feature transformation.
    - In the second example, input channels were compressed from 32 down to 5.
  - As we'll see, this channel reduction can dramatically cut computation costs.
  - When the number of 1 x 1 filters equals the input channel count, the output preserves the same number of channels. In this case, the 1 x 1 conv functions as a nonlinear transformation, learning a pointwise nonlinearity.

- 1 x 1 convolutions can replace fully connected layers, as Yann LeCun has argued they're equivalent.

  - > In Convolutional Nets, there is no such thing as "fully-connected layers". There are only convolution layers with 1x1 convolution kernels and a full connection table. [Yann LeCun](https://www.facebook.com/yann.lecun/posts/10152820758292143) 

- [[Lin et al., 2013. Network in network]](https://arxiv.org/abs/1312.4400)

### 💡 The Inception Module Concept

- When designing a CNN, you face many choices at each layer: 3 x 3 conv, 5 x 5 conv, max pooling, etc.
- The **Inception** philosophy asks: why not try all of them simultaneously?
- **Inception module**, naive version:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//13.png)
  - Note: the max-pool uses "same" padding here.
  - Input dimensions: 28 x 28 x 192; Output dimensions: 28 x 28 x 256.
  - By applying every possible operation, we let the network decide which transformations are most valuable.
  - [[Szegedy et al. 2014. Going deeper with convolutions]](https://arxiv.org/abs/1409.4842)
- The computational cost issue with the Inception module:
  - Focus on just the 5 x 5 Conv from the above example.
  - There are 32 same filters of 5 x 5, and the input is 28 x 28 x 192.
  - Output: 28 x 28 x 32
  - Total multiplications required:
    - Number of outputs × Filter size × Filter size × Input depth
    - Which equals: `28 * 28 * 32 * 5 * 5 * 192 = 120 Mil` 
    - 120 million multiply operations is significant even on modern hardware.
  - By inserting a 1 x 1 convolution, we can slash this from 120M to roughly 12M. Here's the approach.
- Reducing computational cost with 1 x 1 convolution:
  - The revised architecture:
    - X0 shape is (28, 28, 192)
    - Apply 16 (1 x 1 Convolution)
    - This yields X1 of shape (28, 28, 16)
      - Notice: the channel dimension has been compressed.
    - Then apply 32 (5 x 5 Convolution)
    - This yields X2 of shape (28, 28, 32)
  - Multiplication count breakdown:
    - First Conv: `28 * 28 * 16 * 1 * 1 * 192 = 2.5 Mil`
    - Second Conv: `28 * 28 * 32 * 5 * 5 * 16 = 10 Mil`
    - Total: approximately 12.5 Mil — a dramatic improvement over 120 Mil.
- This 1 x 1 Conv layer is referred to as a Bottleneck (`BN`).
- Empirically, the bottleneck layer does not harm accuracy.
- **Inception module**, dimensionality reduction version:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//14.png)
- Inception module implementation example in Keras:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//inception_block1a.png)

### 🌐 The Inception Network (GoogLeNet)

- The Inception network chains together multiple Inception modules.
- The name "Inception" was inspired by a *meme* from the **Inception movie**.
- Complete architecture:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//15.png)
- A Max-Pool block is sometimes placed before an inception module to reduce input dimensionality.
- Three softmax branches appear at different network depths to guide intermediate representations toward the learning objective. The auxiliary outputs (softmax0, softmax1) also serve as regularizers.
- Since the original Inception, multiple subsequent versions have been developed — v2, v3, and v4. Hybrid architectures combining Inception modules with ResNet connections also exist.
- [[Szegedy et al., 2014, Going Deeper with Convolutions]](https://arxiv.org/abs/1409.4842)

### 📂 Leveraging Open-Source Code

- We've explored many NN and ConvNet architectures so far.
- Reproducing these networks from scratch is challenging because papers may omit crucial implementation details. Additional hurdles include:
  - Learning rate schedules.
  - Hyperparameter tuning specifics.
- Fortunately, many deep learning researchers share their implementations publicly on platforms like [Github](Github.com).
- When you encounter an interesting paper, your first step should be to search for an existing open-source implementation.
- A major benefit: you can often download both the architecture and its pretrained weights. The authors may have trained on multiple GPUs for weeks — and the results are available to you immediately.

### 🔄 Transfer Learning Strategies

- When a particular architecture has already been trained, you can leverage those pretrained weights instead of initializing randomly for your specific task.
- This approach can significantly boost performance.
- These pretrained models were likely trained on massive datasets such as ImageNet, MS COCO, or Pascal — requiring enormous time and compute resources. Reusing their weights saves considerable effort.
- Scenario 1 — limited data:
  - Suppose you're classifying cats into 3 categories: Tigger, Misty, and neither.
  - You have insufficient data to train a network from scratch on these https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/.
  - Andrew recommends downloading a well-performing network with its weights, removing the final softmax layer, adding your own output layer, and training only that new layer while keeping all preceding layers frozen.
  - Frameworks provide options to freeze parameters in specific layers using `trainable = 0` or `freeze = 0`.
  - A helpful speedup trick: run the pretrained network (minus the final softmax) to generate intermediate representations for your https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ and save them to disk. Then train a lightweight network on top of these cached features. This bypasses the cost of forward-passing through all layers repeatedly.
    - Essentially you're converting your https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ into fixed-length vectors.
- Scenario 2 — moderate data:
  - If you have a reasonably large dataset, freeze only the early layers of the pretrained network and train the remaining layers.
  - Alternatively, discard the unfrozen layers entirely and replace them with your own custom architecture.
- Scenario 3 — abundant data:
  - With enough data, fine-tune the entire pretrained network — but don't reinitialize the weights randomly. Start from the learned parameters and continue training.

### 🖼️ Expanding Data with Augmentation

- More data generally leads to better performance. Data augmentation is a widely used technique for enhancing deep NN results.
- Computer vision applications almost always benefit from additional data.
- Popular data augmentation techniques for vision tasks:
  - Mirroring (horizontal flipping).
  - Random cropping.
    - Risk: crops may capture irrelevant regions.
    - Mitigation: ensure crops are sufficiently large.
  - Rotation.
  - Shearing.
  - Local warping.
  - Color shifting.
    - For example, apply small perturbations to the R, G, and B channels. The resulting image still looks the same to humans but appears different to the model.
    - Perturbation values are typically drawn from a probability distribution and kept small.
    - This improves robustness to lighting and color variations in https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/. 
    - The ***PCA color augmentation*** algorithm can automatically determine appropriate color shifts.
- Implementation tip for training:
  - Use a separate CPU thread to produce augmented mini-batches while the GPU trains the network.
- Data augmentation introduces its own hyperparameters. A practical starting point is to adopt an open-source augmentation pipeline and then customize it.

### 📊 Current Landscape of Computer Vision

- Different problems sit at different points on the data availability spectrum.
- Speech recognition benefits from vast datasets; image recognition has moderate amounts; object detection typically has less data.
- With abundant data, researchers gravitate toward:
  - Simpler algorithms.
  - Less manual feature engineering.
- With limited data, practitioners often resort to more hand-crafted approaches — including more complex architectures.
- Since many computer vision tasks still lack sufficient data, hand engineering remains important.
- The following chapter introduces more sophisticated architectures for object detection, partly because of the relative data scarcity.
- Strategies for excelling on benchmarks and competitions:
  - Ensembling.
    - Train multiple models independently and average their predictions.
    - After identifying the best architecture, create several randomly initialized copies and train them separately.
    - Typical improvement: around 2%.
    - Drawback: inference slows proportionally, and memory usage multiplies.
    - Common in competitions but rarely deployed in production.
  - Multi-crop at test time.
    - Evaluate the classifier on multiple augmented versions of each test image and average the results.
    - The "10 crops" technique exemplifies this approach.
    - Can improve production results.
- Practical advice:
  - Adopt published network architectures from the literature.
  - Use open-source implementations when available.
  - Start with pretrained models and fine-tune on your specific dataset.

## 🎯 Detecting Objects in Images

> Discover how to apply CNN knowledge to one of the most challenging and dynamic areas of computer vision: object detection.

### 📍 Locating Objects in Images

- Object detection has seen remarkable progress through deep learning in recent years.

- Understanding the different levels of visual understanding:

  - **Image Classification**: 
    - Assign a single class label to the entire image. No spatial information about object location is needed. Typically involves one object per image.
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//Classification.jpg)
  - **Classification with Localization**:
    - Predict both the class label and the bounding box coordinates. Locate the single object with a rectangle. Usually one object per image.
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//ClassificationLoc.jpg)
  - **Object Detection**:
    - Identify all objects from specific classes within an image, providing bounding boxes for each. Multiple objects of different classes may be present.
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//ObjectDetection.png)
  - **Semantic Segmentation**:
    - Assign a category label to every pixel. Only pixel-level classification matters — individual object instances aren't distinguished.
    - If two objects of the same class overlap, they can't be separated.
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//SemanticSegmentation.png)
  - **Instance Segmentation**:
    - The most comprehensive task. Like semantic segmentation but also differentiating between individual instances of the same class.
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//InstanceSegmentation.png)

- For basic image classification, attach a Softmax to the end of a ConvNet.

- For classification with localization, add four additional outputs — `bx`, `by`, `bh`, and `bw` — specifying the object's bounding box. The training data must include these coordinates.

- Structure of the target label Y for classification with localization: 

  - ```
    Y = [
      		Pc				# Probability of an object is presented
      		bx				# Bounding box
      		by				# Bounding box
      		bh				# Bounding box
      		bw				# Bounding box
      		c1				# The classes
      		c2
      		...
    ]
    ```

  - Example (Object is present):

    - ```
      Y = [
        		1		# Object is present
        		0
        		0
        		100
        		100
        		0
        		1
        		0
      ]
      ```

  - Example (No object present):

    - ```
      Y = [
        		0		# Object isn't presented
        		?		# ? means we dont care with other values
        		?
        		?
        		?
        		?
        		?
        		?
      ]
      ```

- The loss function for our target Y (using squared error as an example):

  - ```
    L(y',y) = {
      			(y1'-y1)^2 + (y2'-y2)^2 + ...           if y1 = 1
      			(y1'-y1)^2						if y1 = 0
    		}
    ```

  - In practice: logistic regression for `pc`, log likelihood loss for class labels, and squared error for bounding box coordinates.

### 📌 Detecting Key Landmarks

- Certain computer vision tasks require outputting specific coordinate points — this is known as **landmark detection**.

- Example: in a face recognition system, you might want to pinpoint eye corners, mouth corners, nose tip, etc. This enables applications like facial pose estimation.

- Y structure for detecting 64 facial landmarks:

  - ```
    Y = [
      		THereIsAface				# Probability of face is presented 0 or 1
      		l1x,
      		l1y,
      		....,
      		l64x,
      		l64y
    ]
    ```

- Another use case: extracting a person's body skeleton via key landmark points for action recognition and related tasks.

- Important: in your labeled dataset, if `l1x, l1y` denotes the left corner of the left eye, this convention must remain consistent across all training examples.

### 🔍 Sliding Window Object Detection

- We'll apply a Conv net to object detection through the sliding window detection approach.
- Example scenario: detecting cars in images.
- Step one: train a ConvNet on tightly cropped car https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ and non-car https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/.
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//18.png)
- Once the ConvNet is trained, apply the sliding window method:
  1. Choose a window (rectangle) size.
  2. Partition the image into overlapping windows of that size. Slide with some stride.
  3. Feed each window through the ConvNet and classify as car or not-car.
  4. Repeat steps 2-3 with different window sizes (larger and smaller).
  5. Collect all windows classified as containing a car.
  6. When multiple windows overlap, keep the one with highest confidence.
- The major drawback: computational expense.
- Before deep learning, practitioners used lightweight hand-crafted linear classifiers with sliding windows, which were computationally cheap. Deep learning models make this approach far more expensive.
- The solution: implement sliding windows using a ***convolutional approach***.
- An alternative strategy is to compress the deep learning model for faster inference.

### ⚡ Conv-Based Sliding Windows

- Converting FC layers to convolutional equivalents (for a 4-class prediction):
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//19.png)
  - The FC layer becomes a Conv layer where the filter dimensions equal the input dimensions.
- **Convolutional implementation of sliding windows**:
  - First, assume your trained ConvNet is fully convolutional (no FC layers):
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//20.png)
  - For a 16 x 16 x 3 test image, the naïve approach would run the ConvNet four separate times (once per 16 x 16 window).
  - The convolutional implementation processes it all at once:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//21.png)
  - Each cell in the output corresponds to one sliding window position from the naïve method. The "blue" cell represents the first window, and so on.
  - This is vastly more efficient because computation is shared across all window positions.
  - A larger example:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//22.png)
  - Here 16 sliding windows share computation through a single forward pass.
  - [[Sermanet et al., 2014, OverFeat: Integrated recognition, localization and detection using convolutional networks]](https://arxiv.org/abs/1312.6229)
- A limitation of this method: predicted bounding box positions may not align precisely with objects.
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//23.png)
  - Red: the desired bounding box; Blue: what the algorithm actually captures.

### 📦 Predicting Bounding Boxes

- A superior approach to the previous section is the [YOLO algorithm](https://arxiv.org/abs/1506.02640).

- YOLO stands for *You Only Look Once*, introduced in 2015.

- YOLO Algorithm overview:

  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//24.png)

  1. Consider a 100 × 100 image.
  2. Overlay a 3 x 3 grid (use 19 x 19 for finer results on a 100 x 100 image).
  3. Run the classification-with-localization algorithm on each cell. `bx` and `by` represent the object center relative to the cell (range 0–1), while `bh` and `bw` represent height and width (can exceed 1.0).
  4. Process everything simultaneously using the convolutional sliding window technique. With Y shaped as 1 x 8, the output for a 100 x 100 image is 3 x 3 x 8 (9 cells).
  5. Merge predictions based on predicted center points.

- Challenge: what happens when multiple objects fall in the same grid cell?

- A key strength of YOLO: exceptional speed combined with a fully convolutional implementation.

- What distinguishes YOLO from other detectors? It uses a single CNN for both classification and bounding box localization.

- Subsequent sections introduce refinements that further improve YOLO.

### 📏 Intersection Over Union (IoU)

- Intersection Over Union is a metric for evaluating object detection quality.
- It calculates the area of overlap between two bounding boxes divided by their combined area. In essence, *IoU measures how much two bounding boxes overlap*.
- Illustration:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//25.png)
  - Red: ground truth; Purple: prediction.
  - Compute the union area (both rectangles combined), then the intersection area.
  - `IOU = intersection area / Union area`
- An IoU ≥ 0.5 is typically considered good. The ideal value is 1.
- Higher IoU indicates more precise localization.

### 🚫 Non-Maximum Suppression

- A known issue with YOLO: it can produce multiple detections for a single object.
- Non-max suppression ensures each object receives only one detection.
- Illustration:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//26.png)
  - Each car may trigger two or more detection boxes with varying confidence scores, because multiple grid cells may believe they contain the object's center.
- Non-max suppression procedure:
  1. Assume we're detecting a single class.
  2. Each prediction has the form `[Pc, bx, by, bh, hw]` where Pc is the objectness probability.
  3. Discard all boxes where `Pc < 0.6`.
  4. While boxes remain:
     1. Select the box with highest Pc and output it.
     2. Remove all remaining boxes with `IoU > 0.5` relative to the selected box (i.e., boxes with high overlap exceeding the 0.5 threshold).
- For multi-class detection with `c` classes, run non-max suppression independently for each class.

### ⚓ Using Anchor Boxes

- In standard YOLO, each grid cell detects at most one object. But what if a cell contains multiple objects?
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//27.png)
  - Here a car and a person share the same cell.
  - In practice, this situation is rare.
- Anchor boxes address this limitation.
- With a single anchor: Y = `[Pc, bx, by, bh, bw, c1, c2, c3]`. With two anchors:
  - Y = `[Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3]` — simply duplicating the structure.
  - Each anchor box has a predefined shape:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//28.png)
- Previously: each training object was assigned to the grid cell containing its midpoint.
- With anchor boxes: each object is assigned to both the grid cell containing its midpoint AND the anchor box with the <u>highest IoU</u>. Object assignment is based on which anchor box shape most closely matches the object's bounding box.
- Data example:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//29.png)
  - The car better matches anchor 2 than anchor 1.
- You can define two or more anchor boxes; their shapes must be predetermined.
  - Traditionally, anchor shapes are selected manually — perhaps 5 or 10 shapes that cover the range of objects you expect to detect.
  - K-means clustering on your dataset can also determine good anchor shapes automatically.
- Anchor boxes enable the algorithm to specialize — for instance, making it easier to detect wider https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ versus taller ones.

### 🎯 The Complete YOLO Pipeline

- YOLO is a cutting-edge object detection system prized for both speed and accuracy.

- Let's consolidate everything into the full YOLO algorithm with an example.

- Imagine building object detection for an autonomous driving system. It must identify three classes:

  1. Pedestrian.
  2. Car.
  3. Motorcycle.

- We select two anchor boxes — one tall, one wide.

  - In practice, five or more anchors are used, either hand-designed or generated via k-means.

- The labeled Y tensor has shape `[Ny, HeightOfGrid, WidthOfGrid, 16]`, where Ny is the number of samples and each 16-element row is:

  - `[Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3]`

- Your dataset consists of images with multiple labels and bounding rectangles. Transform these into the agreed Y format.

  - An example:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//30.png)
  - Initialize everything to zeros and ?, then for each label and rectangle: find the closest grid cell, determine the shape, and select the best anchor via IoU. The resulting Y shape per image is `[HeightOfGrid, WidthOfGrid, 16]`.

- Train the labeled https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ on a Conv net. Expected output shape: `[HeightOfGrid, WidthOfGrid, 16]`.

- At inference: run the ConvNet on a test image, then apply non-max suppression separately for each of the 3 classes.

  - Initial detections might look like:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//31.png)
    - Total generated boxes = grid_width × grid_height × number_of_anchors = 3 × 3 × 2
  - After filtering out low-confidence predictions:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//32.png)
  - After selecting highest-confidence boxes and applying IoU-based filtering:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//33.png)

- YOLO tends to struggle with very small objects.

- [YOLO9000 Better, faster, stronger](https://arxiv.org/abs/1612.08242)

  - Summary:

  - ```
    ________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                
    ========================================================================================
    input_1 (InputLayer)             (None, 608, 608, 3)   0                                 
    ________________________________________________________________________________________
    conv2d_1 (Conv2D)                (None, 608, 608, 32)  864         input_1[0][0]         
    ________________________________________________________________________________________
    batch_normalization_1 (BatchNorm (None, 608, 608, 32)  128         conv2d_1[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)        (None, 608, 608, 32)  0     batch_normalization_1[0][0] 
    ________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)   (None, 304, 304, 32)  0           leaky_re_lu_1[0][0]   
    ________________________________________________________________________________________
    conv2d_2 (Conv2D)                (None, 304, 304, 64)  18432       max_pooling2d_1[0][0] 
    ________________________________________________________________________________________
    batch_normalization_2 (BatchNorm (None, 304, 304, 64)  256         conv2d_2[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)        (None, 304, 304, 64)  0     batch_normalization_2[0][0] 
    _______________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)   (None, 152, 152, 64)  0           leaky_re_lu_2[0][0]   
    ________________________________________________________________________________________
    conv2d_3 (Conv2D)                (None, 152, 152, 128) 73728       max_pooling2d_2[0][0] 
    ________________________________________________________________________________________
    batch_normalization_3 (BatchNorm (None, 152, 152, 128) 512         conv2d_3[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)        (None, 152, 152, 128) 0     batch_normalization_3[0][0] 
    ________________________________________________________________________________________
    conv2d_4 (Conv2D)                (None, 152, 152, 64)  8192        leaky_re_lu_3[0][0]   
    ________________________________________________________________________________________
    batch_normalization_4 (BatchNorm (None, 152, 152, 64)  256         conv2d_4[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)        (None, 152, 152, 64)  0     batch_normalization_4[0][0] 
    ________________________________________________________________________________________
    conv2d_5 (Conv2D)                (None, 152, 152, 128) 73728       leaky_re_lu_4[0][0]   
    ________________________________________________________________________________________
    batch_normalization_5 (BatchNorm (None, 152, 152, 128) 512         conv2d_5[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)        (None, 152, 152, 128) 0     batch_normalization_5[0][0] 
    ________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)   (None, 76, 76, 128)   0           leaky_re_lu_5[0][0]   
    ________________________________________________________________________________________
    conv2d_6 (Conv2D)                (None, 76, 76, 256)   294912      max_pooling2d_3[0][0] 
    _______________________________________________________________________________________
    batch_normalization_6 (BatchNorm (None, 76, 76, 256)   1024        conv2d_6[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)        (None, 76, 76, 256)   0     batch_normalization_6[0][0] 
    _______________________________________________________________________________________
    conv2d_7 (Conv2D)                (None, 76, 76, 128)   32768       leaky_re_lu_6[0][0]   
    ________________________________________________________________________________________
    batch_normalization_7 (BatchNorm (None, 76, 76, 128)   512         conv2d_7[0][0]       
    _______________________________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)        (None, 76, 76, 128)   0     batch_normalization_7[0][0] 
    ________________________________________________________________________________________
    conv2d_8 (Conv2D)                (None, 76, 76, 256)   294912      leaky_re_lu_7[0][0]   
    ________________________________________________________________________________________
    batch_normalization_8 (BatchNorm (None, 76, 76, 256)   1024        conv2d_8[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)        (None, 76, 76, 256)   0     batch_normalization_8[0][0] 
    ________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)   (None, 38, 38, 256)   0           leaky_re_lu_8[0][0]   
    ________________________________________________________________________________________
    conv2d_9 (Conv2D)                (None, 38, 38, 512)   1179648     max_pooling2d_4[0][0] 
    ________________________________________________________________________________________
    batch_normalization_9 (BatchNorm (None, 38, 38, 512)   2048        conv2d_9[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)        (None, 38, 38, 512)   0     batch_normalization_9[0][0] 
    ________________________________________________________________________________________
    conv2d_10 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_9[0][0]   
    ________________________________________________________________________________________
    batch_normalization_10 (BatchNor (None, 38, 38, 256)   1024        conv2d_10[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)       (None, 38, 38, 256)   0    batch_normalization_10[0][0]
    ________________________________________________________________________________________
    conv2d_11 (Conv2D)               (None, 38, 38, 512)   1179648    leaky_re_lu_10[0][0]   
    ________________________________________________________________________________________
    batch_normalization_11 (BatchNor (None, 38, 38, 512)   2048        conv2d_11[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)       (None, 38, 38, 512)   0    batch_normalization_11[0][0]
    _______________________________________________________________________________________
    conv2d_12 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_11[0][0] 
    ________________________________________________________________________________________
    batch_normalization_12 (BatchNor (None, 38, 38, 256)   1024        conv2d_12[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)       (None, 38, 38, 256)   0   batch_normalization_12[0][0]
    ________________________________________________________________________________________
    conv2d_13 (Conv2D)               (None, 38, 38, 512)   1179648     leaky_re_lu_12[0][0] 
    ________________________________________________________________________________________
    batch_normalization_13 (BatchNor (None, 38, 38, 512)   2048        conv2d_13[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)       (None, 38, 38, 512)   0    batch_normalization_13[0][0]
    ________________________________________________________________________________________
    max_pooling2d_5 (MaxPooling2D)   (None, 19, 19, 512)   0           leaky_re_lu_13[0][0] 
    _______________________________________________________________________________________
    conv2d_14 (Conv2D)               (None, 19, 19, 1024)  4718592     max_pooling2d_5[0][0] 
    ________________________________________________________________________________________
    batch_normalization_14 (BatchNor (None, 19, 19, 1024)  4096        conv2d_14[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)       (None, 19, 19, 1024)  0    batch_normalization_14[0][0]
    ________________________________________________________________________________________
    conv2d_15 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_14[0][0] 
    ________________________________________________________________________________________
    batch_normalization_15 (BatchNor (None, 19, 19, 512)   2048        conv2d_15[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_15 (LeakyReLU)       (None, 19, 19, 512)   0    batch_normalization_15[0][0]
    ________________________________________________________________________________________
    conv2d_16 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_15[0][0] 
    ________________________________________________________________________________________
    batch_normalization_16 (BatchNor (None, 19, 19, 1024)  4096        conv2d_16[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_16 (LeakyReLU)       (None, 19, 19, 1024)  0    batch_normalization_16[0][0]
    ________________________________________________________________________________________
    conv2d_17 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_16[0][0] 
    ________________________________________________________________________________________
    batch_normalization_17 (BatchNor (None, 19, 19, 512)   2048        conv2d_17[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_17 (LeakyReLU)       (None, 19, 19, 512)   0    batch_normalization_17[0][0]
    _______________________________________________________________________________________
    conv2d_18 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_17[0][0] 
    ________________________________________________________________________________________
    batch_normalization_18 (BatchNor (None, 19, 19, 1024)  4096        conv2d_18[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_18 (LeakyReLU)       (None, 19, 19, 1024)  0    batch_normalization_18[0][0]
    ________________________________________________________________________________________
    conv2d_19 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_18[0][0] 
    ________________________________________________________________________________________
    batch_normalization_19 (BatchNor (None, 19, 19, 1024)  4096        conv2d_19[0][0]       
    ________________________________________________________________________________________
    conv2d_21 (Conv2D)               (None, 38, 38, 64)    32768       leaky_re_lu_13[0][0]
    ________________________________________________________________________________________
    leaky_re_lu_19 (LeakyReLU)       (None, 19, 19, 1024)  0    batch_normalization_19[0][0]
    ________________________________________________________________________________________
    batch_normalization_21 (BatchNor (None, 38, 38, 64)    256         conv2d_21[0][0]       
    ________________________________________________________________________________________
    conv2d_20 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_19[0][0]
    ________________________________________________________________________________________
    leaky_re_lu_21 (LeakyReLU)       (None, 38, 38, 64)    0    batch_normalization_21[0][0]
    ________________________________________________________________________________________
    batch_normalization_20 (BatchNor (None, 19, 19, 1024)  4096        conv2d_20[0][0]       
    ________________________________________________________________________________________
    space_to_depth_x2 (Lambda)       (None, 19, 19, 256)   0           leaky_re_lu_21[0][0] 
    ________________________________________________________________________________________
    leaky_re_lu_20 (LeakyReLU)       (None, 19, 19, 1024)  0    batch_normalization_20[0][0]
    ________________________________________________________________________________________
    concatenate_1 (Concatenate)      (None, 19, 19, 1280)  0         space_to_depth_x2[0][0] 
                                                                      leaky_re_lu_20[0][0] 
    ________________________________________________________________________________________
    conv2d_22 (Conv2D)               (None, 19, 19, 1024)  11796480    concatenate_1[0][0]   
    ________________________________________________________________________________________
    batch_normalization_22 (BatchNor (None, 19, 19, 1024)  4096        conv2d_22[0][0]       
    ________________________________________________________________________________________
    leaky_re_lu_22 (LeakyReLU)       (None, 19, 19, 1024)  0    batch_normalization_22[0][0]
    ________________________________________________________________________________________
    conv2d_23 (Conv2D)               (None, 19, 19, 425)   435625      leaky_re_lu_22[0][0] 
    ===============================================================================================
    Total params: 50,983,561
    Trainable params: 50,962,889
    Non-trainable params: 20,672
    _______________________________________________________________________________________________
    ```

- YOLO implementation resources:

  - https://github.com/allanzelener/YAD2K
  - https://github.com/thtrieu/darkflow
  - https://pjreddie.com/darknet/yolo/

### 🔲 Region-Based CNNs (R-CNN Family)

- R-CNN is an alternative approach to object detection.

- On YOLO's speed advantage:

  - > Our model has several advantages over classifier-based systems. It looks at the whole image at test time so its predictions are informed by global context in the image. It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. This makes it extremely fast, more than 1000x faster than R-CNN and 100x faster than Fast R-CNN. See our paper for more details on the full system.

- However, YOLO does waste computation on many regions devoid of objects.

- **R-CNN** = Regions with Convolutional Neural Networks.

- R-CNN selectively identifies promising regions and runs a ConvNet classifier on each one.

- The region selection method is a segmentation algorithm that produces results like:

  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//34.png)

- If the segmentation yields 2000 regions, the classifier/CNN evaluates each one individually.

- The R-CNN family has evolved significantly toward faster inference:

  - R-CNN:
    - Propose regions. Classify each region independently. Output: label + bounding box.
    - Drawback: very slow.
    - [[Girshik et. al, 2013. Rich feature hierarchies for accurate object detection and semantic segmentation]](https://arxiv.org/abs/1311.2524)
  - Fast R-CNN:
    - Propose regions. Use a convolutional sliding window to classify all regions simultaneously.
    - [[Girshik, 2015. Fast R-CNN]](https://arxiv.org/abs/1504.08083)
  - Faster R-CNN:
    - Use a convolutional network to generate region proposals.
    - [[Ren et. al, 2016. Faster R-CNN: Towards real-time object detection with region proposal networks]](https://arxiv.org/abs/1506.01497)
  - Mask R-CNN:
    - https://arxiv.org/abs/1703.06870

- Even Faster R-CNN implementations are generally slower than YOLO.

- Andrew Ng favors the YOLO philosophy because it accomplishes everything in a single pass rather than two separate stages.

- Other single-shot detection architectures include **SSD** and **MultiBox**.

  - [[Wei Liu, et. al 2015 SSD: Single Shot MultiBox Detector]](https://arxiv.org/abs/1512.02325)

- **R-FCN** resembles Faster R-CNN but is more computationally efficient.

  - [[Jifeng Dai, et. al 2016 R-FCN: Object Detection via Region-based Fully Convolutional Networks ]](https://arxiv.org/abs/1605.06409)

## 🎭 Advanced Applications: Faces & Artistic Style

> Explore how CNNs extend beyond classification to creative and biometric applications. Build your own algorithms for art generation and face recognition!

### 👤 Recognizing Faces

#### 🤷 Face Recognition Explained

- A face recognition system identifies individuals from their facial features. It operates on both https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ or videos.
- **<u>Liveness detection</u>** in video-based face recognition prevents the system from being fooled by a static photograph. It can be trained via supervised deep learning using datasets of live and non-live subjects, potentially combined with sequence models.
- Face verification vs. face recognition:
  - Verification:
    - Input: image + name/ID. (1 : 1)
    - Output: is this really the claimed person?
    - The question: "Is this the person they claim to be?"
  - Recognition:
    - Maintains a database of K individuals.
    - Receives an input image.
    - Output: the person's ID if they match someone in the database (or "not recognized").
    - The question: "Who is this person?"
- A face verification module can serve as the foundation for a face recognition system. However, verification accuracy must be extremely high (~99.9%+) because recognition accuracy degrades across K individuals.

#### ☝️ Learning from a Single Example

- A core challenge in face recognition is the one-shot learning problem.
- One-Shot Learning: successfully recognizing a person after seeing just one training image.
- Deep learning traditionally requires large datasets to perform well.
- The workaround is to learn a **similarity function**:
  - d( **img1**, **img2** ) = degree of difference between https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/.
  - We want d to be small when the two faces belong to the same person.
  - We define a threshold tau T:
    - If d( **img1**, **img2** ) <= T    → same person.
- The similarity function elegantly solves one-shot learning and generalizes well to new individuals.

#### 🔀 The Siamese Architecture

- We implement the similarity function through Siamese Networks — architectures where two or more networks share identical parameters and process multiple inputs simultaneously.
- Siamese network structure:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//35.png)
  - Two identical ConvNets encode input images into embedding vectors. In the diagram, each vector has shape (128, ).
  - Loss function: `d(x1, x2) = || f(x1) - f(x2) ||^2`
  - Goal: d should be small for same-person pairs and large for different-person pairs.
  - [[Taigman et. al., 2014. DeepFace closing the gap to human level performance]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)

#### 🔺 Triplet Loss Function

- Triplet Loss is a loss function designed for training the similarity distance in Siamese networks.
- The learning objective: ensure the distance between an **Anchor** image and a **Positive** image is smaller than between the Anchor and a **Negative** image.
  - Positive = same individual; Negative = different individual.
- The name "triplet" reflects comparing three images: Anchor (A), Positive (P), and Negative (N).
- Formal requirements:
  - Positive distance must be smaller than negative distance:
  - `||f(A) - f(P)||^2  <= ||f(A) - f(N)||^2`
  - Equivalently:
  - `||f(A) - f(P)||^2  - ||f(A) - f(N)||^2 <= 0`
  - To prevent trivial solutions (e.g., all-zero embeddings):
  - `||f(A) - f(P)||^2  - ||f(A) - f(N)||^2 <= -alpha`
    - Alpha is a small positive constant, often called the margin.
  - Rearranging:
  - `||f(A) - f(P)||^2  - ||f(A) - f(N)||^2 + alpha <= 0`
- Complete loss function:
  - Given 3 https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ (A, P, N):
  - `L(A, P, N) = max (||f(A) - f(P)||^2  - ||f(A) - f(N)||^2 + alpha , 0)`
  - `J = Sum(L(A[i], P[i], N[i]) , i)` across all triplets of https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/.
- Your dataset must contain multiple https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ per person. Generate triplets from this dataset, which must be sufficiently large.
- Selecting triplets A, P, N:
  - Random selection (where A and P share an identity and A and N don't) makes the constraint too easy to satisfy:
    - `d(A, P) + alpha <= d (A, N)` 
    - The network learns little from such easy examples.
  - Instead, deliberately choose **hard** triplets for training:
    - Target triplets where: `d(A, P) + alpha <= d (A, N)` is nearly violated.
    - This can be achieved using similar poses, for example.
    - See the paper for more details.
- Full details: [[Schroff et al.,2015, FaceNet: A unified embedding for face recognition and clustering]](https://arxiv.org/abs/1503.03832)
- Production face recognition systems train on enormous datasets — 10 to 100+ million https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/.
- Numerous pretrained face recognition models and weights are publicly available.

#### ✅ Binary Classification for Face Verification

- Beyond triplet loss, face similarity can also be learned as a binary classification problem.
- Alternative similarity learning approach:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//36.png)
  - The output layer uses a sigmoid activation.
  - `Y' = wi * Sigmoid ( f(x(i)) - f(x(j)) ) + b` where the subtraction computes the Manhattan distance between f(x(i)) and f(x(j)).
  - Alternative distance metrics include Euclidean and chi-square similarity.
  - The architecture is Siamese — both branches share identical parameters.
- Reference: [[Taigman et. al., 2014. DeepFace closing the gap to human level performance]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)
- Deployment optimization:
  - Pre-compute embeddings f(x(j)) for all reference https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ and store them.
  - For each new image, compute f(x(i)), compare against all stored embeddings, and pass pairs through the sigmoid.
- This approach performs comparably to triplet loss.
- Available face recognition implementations using deep learning:
  - [Openface](https://cmusatyalab.github.io/openface/)
  - [FaceNet](https://github.com/davidsandberg/facenet)
  - [DeepFace](https://github.com/RiweiChen/DeepFace)

### 🎨 Artistic Neural Style Transfer

#### 🖌️ What is Neural Style Transfer?

- Neural style transfer is a fascinating application of convolutional networks.
- It takes a content image `C` and a style image `S`, producing a generated image `G` that combines the content of C with the artistic style of S.
- ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//37.png)
- Implementation requires examining features extracted at both shallow and deep layers of the ConvNet.
- It builds upon a pretrained convolutional network (like VGG). Repurposing a network trained on a different task is called transfer learning.

#### 🧠 Visualizing What ConvNets Learn

- Exploring what a deep network has learned:
  - Given an AlexNet-style ConvNet:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//38.png)
  - Select a neuron in layer l. Find the nine image patches that produce the highest activation for that neuron.
    - Note: neurons in early layers have small receptive fields, so their visualizations correspond to small image patches. Deeper layers have larger receptive fields and produce larger visual patterns.
  - Repeat across different neurons and layers.
  - Layer 1 typically captures low-level features like colors and edges.
- Each successive layer learns increasingly complex and abstract representations.
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//39.png)
- Layer 1 visualizations are derived from the first layer's weights. Deeper layer https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ are generated by finding the receptive field regions that maximally activate each neuron.
- [[Zeiler and Fergus., 2013, Visualizing and understanding convolutional networks]](https://arxiv.org/abs/1311.2901)
- A helpful resource on computing **receptive fields** for a given layer:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//receptiveField.png)
  - From [A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

#### 💰 The Overall Cost Function

- We define a cost function for the generated image that quantifies its quality.
- Given content image C, style image S, and generated image G:
  - `J(G) = alpha * J(C,G) + beta * J(S,G)`
  - `J(C, G)` captures how closely the generated image resembles the content image.
  - `J(S, G)` captures how closely the generated image matches the style image.
  - alpha and beta control the relative influence of content vs. style — they are hyperparameters.
- Procedure for finding G:
  1. Initialize G randomly
     - For example G: 100 X 100 X 3
  2. Minimize `J(G)` using gradient descent
     - `G = G - dG`  Compute the gradient of the cost with respect to the image pixels and update iteratively.
- The iterative process looks like:
  - Target output:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//40.png)
  - Progressive refinement:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//41.png)

#### 📝 Content Similarity Cost

- We previously established that we need separate cost functions for content and style similarity.
- Choose a hidden layer `l` for computing the content cost. 
  - If `l` is very shallow (e.g., layer 1), the generated image will be forced to closely replicate the original content.
  - In practice, `l` is chosen at an intermediate depth — neither too shallow nor too deep.
- Use a pre-trained ConvNet (e.g., VGG network).
- Let `a(c)[l]` and `a(G)[l]` represent the activations at layer `l` for the https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/.
- When `a(c)[l]` and `a(G)[l]` are similar, the images share the same content:
  - `J(C, G) at a layer l = 1/2 || a(c)[l] - a(G)[l] ||^2`

#### 🎭 Style Similarity Cost

- Defining the ***style*** of an image:
  - Consider using layer l's activations to characterize ***style***.
  - Style is defined as the correlation between **activations** across different **channels**.
    - Conceptually:
      - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//42.png)
    - How correlated is the orange channel with the yellow channel?
    - Correlated: when a certain pattern appears in one channel, a specific pattern tends to appear in the other (they co-occur).
    - Uncorrelated: a pattern in one channel tells us nothing about what appears in another.
  - Correlation captures which visual components tend to co-occur within an image.
- The channel correlations from the style image should be reproduced in the generated image.
- Style matrix (Gram matrix):
  - Let `a(l)[i, j, k]` denote the activation at layer l at position `(i=H, j=W, k=C)`.
  - Also `G(l)(s)` is a matrix of shape `nc(l) x nc(l)`.
    - Called the style matrix or Gram matrix.
    - Each entry quantifies the correlation between a pair of channels.
  - The Gram matrices are computed for both the style image and the generated image using:
    - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//43.png)
    - Each entry equals the sum of element-wise products across spatial positions.
- Efficient Gram matrix computation:
  - Reshape the activation tensor from H × W × C to HW × C.
  - Call this reshaped tensor F.
  - `G[l] = F * F.T`
- The style cost function:
  - `J(S, G) at layer l = (1/ 2 * H * W * C) || G(l)(s) - G(l)(G) ||`
- When aggregating across multiple layers:
  - `J(S, G) = Sum (lamda[l]*J(S, G)[l], for all layers)`
- Steps for building a TensorFlow neural style transfer model:
  1. Create an Interactive Session.
  2. Load the content image.
  3. Load the style image.
  4. Randomly initialize the generated image.
  5. Load the VGG16 model.
  6. Build the TensorFlow computation graph:
     - Forward-pass the content image through VGG16 and compute content cost.
     - Forward-pass the style image through VGG16 and compute style cost.
     - Compute total cost.
     - Define the optimizer and learning rate.
  7. Initialize the graph and iterate, updating the generated image at each step.

#### 📐 Extending Convolutions to 1D and 3D

- So far, convolutions have been applied to https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images/ which are inherently 2D.
- Convolutions generalize naturally to 1D and 3D data.
- 1D convolution example:
  - Input shape (14, 1)
  - Apply 16 filters with F = 5, S = 1
  - Output shape: 10 × 16
  - Apply 32 filters with F = 5, S = 1
  - Output shape: 6 × 32
- The standard formula `(N - F)/S + 1` applies here, producing a vector instead of a 2D matrix.
- 1D data appears in many domains: audio waveforms, sound signals, heartbeat monitors.
- For most 1D sequential data, Recurrent Neural Networks (RNNs) are typically preferred.
- 3D data arises in applications such as CT scans:
  - ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/4-%20Convolutional%20Neural%20Networks/Images//44.png)
- 3D convolution example:
  - Input shape (14, 14, 14, 1)
  - Apply 16 filters with F = 5, S = 1
  - Output shape (10, 10, 10, 16)
  - Apply 32 filters with F = 5, S = 1
  - Output shape: (6, 6, 6, 32)

## 🧩 Additional Topics

### 🐍 Working with Keras

- Keras is a high-level neural networks API written in Python, capable of running on top of multiple backends including TensorFlow, Theano, and CNTK.
- It was created to help deep learning engineers rapidly prototype and experiment with various model architectures.
- Keras provides an additional abstraction layer above frameworks like TensorFlow, much like TensorFlow sits above raw Python.
- Keras handles many standard model architectures seamlessly.
- Available layer types in Keras:
  - Dense (Fully connected layers).
    - A linear transformation followed by a nonlinear activation.
  - Convolutional layer.
  - Pooling layer.
  - Normalization layer.
    - Implements batch normalization.
  - Flatten layer.
    - Converts a multi-dimensional tensor into a 1D vector.
  - Activation layer.
    - Supported activations: relu, tanh, sigmoid, softmax, and more.
- The four-step Keras workflow for training and evaluation:
  1. Define the model architecture.
  2. Compile: `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
  3. Train: `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`
     - You can optionally include a validation set.
  4. Evaluate: `model.evaluate(x = ..., y = ...)`
- Concise workflow summary: Create → Compile → Fit/Train → Evaluate/Test
- `Model.summary()` provides detailed information about each layer — inputs, outputs, and parameter counts.
- To switch the Keras backend, edit `$HOME/.keras/keras.json` and specify Theano, TensorFlow, or another supported backend.
- After creating a model, you can execute it within a TensorFlow session directly — without going through the compile/train/evaluate pipeline.
- Save models with `model_save` and restore them with `model_load`. This persists the complete architecture and trained weights to disk.
