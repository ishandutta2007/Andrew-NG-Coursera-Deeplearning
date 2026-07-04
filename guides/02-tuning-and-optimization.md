
# ⚙️ Tuning, Regularization & Optimization Techniques

This guide corresponds to the second course in the deep learning specialization on [Coursera](https://www.coursera.org/specializations/deep-learning), hosted by [DeepLearning.ai](http://deeplearning.ai/). Andrew Ng is the instructor.

# 📚 Complete Guide Collection

* [**🧠 Foundations of Neural Networks & Deep Learning**](guides/01-foundations-of-neural-nets.md)
* [**⚙️ Tuning, Regularization & Optimization Techniques**](guides/02-tuning-and-optimization.md)
* [**🎯 ML Project Strategy Playbook**](guides/03-project-strategy-playbook.md)
* [**👁️ Visual Recognition with Convolutional Networks**](guides/04-visual-recognition-networks.md)
* [**🔄 Temporal Sequence Learning Models**](guides/05-temporal-sequence-learning.md)

## 📋 Navigation

* [Tuning, Regularization & Optimization Techniques](#️-tuning-regularization--optimization-techniques)
   * [Navigation](#-navigation)
   * [What This Course Delivers](#-what-this-course-delivers)
   * [Real-World Deep Learning Practices](#-real-world-deep-learning-practices)
      * [Splitting Data: Train / Dev / Test](#-splitting-data-train--dev--test)
      * [Understanding Bias & Variance](#-understanding-bias--variance)
      * [A Practical ML Troubleshooting Guide](#-a-practical-ml-troubleshooting-guide)
      * [Taming Overfitting with Regularization](#-taming-overfitting-with-regularization)
      * [How Does Regularization Prevent Overfitting?](#-how-does-regularization-prevent-overfitting)
      * [The Dropout Technique](#-the-dropout-technique)
      * [Dropout Intuition Deep Dive](#-dropout-intuition-deep-dive)
      * [Additional Regularization Approaches](#-additional-regularization-approaches)
      * [Standardizing Your Inputs](#-standardizing-your-inputs)
      * [The Vanishing & Exploding Gradient Problem](#-the-vanishing--exploding-gradient-problem)
      * [Smart Weight Initialization Strategies](#-smart-weight-initialization-strategies)
      * [Approximating Gradients Numerically](#-approximating-gradients-numerically)
      * [Practical Gradient Checking Tips](#-practical-gradient-checking-tips)
      * [Initialization Recap](#-initialization-recap)
      * [Regularization Recap](#-regularization-recap)
   * [Advanced Optimization Methods](#-advanced-optimization-methods)
      * [Mini-Batch Gradient Descent Explained](#-mini-batch-gradient-descent-explained)
      * [Deeper Look at Mini-Batch Behavior](#-deeper-look-at-mini-batch-behavior)
      * [Exponentially Weighted Moving Averages](#-exponentially-weighted-moving-averages)
      * [Intuition Behind Weighted Averages](#-intuition-behind-weighted-averages)
      * [Correcting Bias in Weighted Averages](#-correcting-bias-in-weighted-averages)
      * [Momentum-Enhanced Gradient Descent](#-momentum-enhanced-gradient-descent)
      * [RMSprop Optimizer](#-rmsprop-optimizer)
      * [The Adam Optimizer](#-the-adam-optimizer)
      * [Scheduling Learning Rate Decay](#-scheduling-learning-rate-decay)
      * [Local Optima in High Dimensions](#-local-optima-in-high-dimensions)
   * [Hyperparameter Search, Batch Norm & Frameworks](#-hyperparameter-search-batch-norm--frameworks)
      * [The Art of Hyperparameter Tuning](#-the-art-of-hyperparameter-tuning)
      * [Picking the Right Scale for Hyperparameters](#-picking-the-right-scale-for-hyperparameters)
      * [Tuning Strategies: Babysitting vs. Parallel Training](#-tuning-strategies-babysitting-vs-parallel-training)
      * [Batch Normalization of Hidden Activations](#-batch-normalization-of-hidden-activations)
      * [Integrating Batch Norm into Your Network](#-integrating-batch-norm-into-your-network)
      * [Why Batch Normalization is So Effective](#-why-batch-normalization-is-so-effective)
      * [Handling Batch Norm During Inference](#-handling-batch-norm-during-inference)
      * [Multi-Class Classification with Softmax](#-multi-class-classification-with-softmax)
      * [Training the Softmax Classifier](#-training-the-softmax-classifier)
      * [Choosing a Deep Learning Framework](#-choosing-a-deep-learning-framework)
      * [Getting Started with TensorFlow](#-getting-started-with-tensorflow)
   * [Supplementary Notes](#-supplementary-notes)

## 🎬 What This Course Delivers

Below is the official description from the [course page](https://www.coursera.org/learn/deep-neural-network):

> This course will teach you the "magic" of getting deep learning to work well. Rather than the deep learning process being a black box, you will understand what drives performance, and be able to more systematically get good results. You will also learn TensorFlow. 
>
> After 3 weeks, you will: 
> - Understand industry best-practices for building deep learning applications. 
> - Be able to effectively use the common neural network "tricks", including initialization, L2 and dropout regularization, Batch normalization, gradient checking, 
> - Be able to implement and apply a variety of optimization algorithms, such as mini-batch gradient descent, Momentum, RMSprop and Adam, and check for their convergence. 
> - Understand new best-practices for the deep learning era of how to set up train/dev/test sets and analyze bias/variance
> - Be able to implement a neural network in TensorFlow. 
>
> This is the second course of the Deep Learning Specialization.



## 🛠️ Real-World Deep Learning Practices

### 📂 Splitting Data: Train / Dev / Test

- Getting every hyperparameter right on the first attempt for a new problem is essentially impossible.
- The workflow follows a cycle: `Idea ==> Code ==> Experiment`.
- Expect to iterate through this loop many times before settling on good hyperparameters.
- Your dataset gets divided into three segments:
  - **Training set** — should be the largest portion
  - **Hold-out / cross-validation / "dev" set** — used for hyperparameter tuning
  - **Testing set** — final evaluation only
- The process: build your model on the training set, then refine hyperparameters using the dev set. Once satisfied, evaluate on the test set.
- Typical data split ratios have evolved over time:
  - Datasets of 100 to 1,000,000 samples ==> 60/20/20
  - Datasets exceeding 1,000,000 samples ==> 98/1/1 or 99.5/0.25/0.25
- Modern practice allocates the lion's share of data to training.
- Critically, the dev and test sets must share the same data distribution.
  - For instance, if training images come from the web but dev/test images come from mobile phones, you'll have a distribution mismatch. Keep dev and test sources consistent.
- The dev set's purpose is to compare your best candidate models.
- Running without a test set (dev set only) is acceptable. Many people call this the "test set," but "dev set" is the more accurate term since it's used during development.

### ⚖️ Understanding Bias & Variance

- Bias and variance concepts are straightforward to learn but tricky to apply well.
- Here's the breakdown:
  - **Underfitting** (e.g., fitting a linear model to non-linear data) indicates **high bias**
  - **Overfitting** indicates **high variance**
  - The sweet spot comes from properly balancing bias and variance
  - Visual reference:
    - ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//01-_Bias_-_Variance.png)
- A numeric way to diagnose bias/variance without plotting:
  - 🔴 High variance (overfitting) scenario:
    - Training error: 1%
    - Dev error: 11%
  - 🔵 High bias (underfitting) scenario:
    - Training error: 15%
    - Dev error: 14%
  - 🟣 High bias AND high variance simultaneously:
    - Training error: 15%
    - Test error: 30%
  - 🟢 Ideal performance:
    - Training error: 0.5%
    - Test error: 1%
  - These benchmarks assume roughly 0% human error. If that assumption doesn't hold, use the human error rate as your baseline instead.

### 🧰 A Practical ML Troubleshooting Guide

- When your model shows **high bias**:
  - Scale up the network (more hidden units, additional layers)
  - Experiment with a different architecture better suited to your data
  - Train for more epochs
  - Explore more sophisticated optimization algorithms
- When your model shows **high variance**:
  - Gather more training data
  - Apply regularization techniques
  - Try a different network design more appropriate for the task
- Keep iterating on both fronts until you achieve low bias AND low variance.
- Before the deep learning era, there was an inherent "bias-variance tradeoff." Today's rich toolbox makes it far more practical to tackle each problem independently.
- Scaling up your neural network almost never hurts performance.

### 🛡️ Taming Overfitting with Regularization

- Incorporating regularization into your NN helps combat variance (overfitting).
- L1 matrix norm:
  
  - `||W|| = Sum(|w[i,j]|)  # sum of absolute values of all w`
- The L2 matrix norm (technically called the Frobenius norm due to linear algebra conventions):
  - `||W||^2 = Sum(|w[i,j]|^2)	# sum of all w squared`
  - Equivalently: `||W||^2 = W.T * W`
- Regularized logistic regression:
  - Standard cost function to minimize: `J(w,b) = (1/m) * Sum(L(y(i),y'(i)))`
  - With L2 regularization: `J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|^2)`
  - With L1 regularization: `J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|)`
  - L1 regularization drives many weight values to exactly zero, producing sparser, more compact models.
  - L2 regularization is used far more frequently in practice.
  - `lambda` is the regularization strength (a hyperparameter you must set).
- Regularized neural networks:
  - The standard cost function:   
    `J(W1,b1...,WL,bL) = (1/m) * Sum(L(y(i),y'(i)))`

  - Adding L2 regularization:   
    `J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum((||W[l]||^2)`

  - We flatten the weight matrix into a single vector `(mn,1)` and compute `sqrt(w1^2 + w2^2.....)`

  - Original backpropagation gradient:   
    `dw[l] = (from back propagation)`

  - Updated gradient with regularization:   
    `dw[l] = (from back propagation) + lambda/m * w[l]`

  - Substituting into the weight update rule:

    - ```
      w[l] = w[l] - learning_rate * dw[l]
           = w[l] - learning_rate * ((from back propagation) + lambda/m * w[l])
           = w[l] - (learning_rate*lambda/m) * w[l] - learning_rate * (from back propagation) 
           = (1 - (learning_rate*lambda)/m) * w[l] - learning_rate * (from back propagation)
      ```

  - Practically speaking, this penalizes large weight values and constrains the model's capacity.

  - The factor `(1 - (learning_rate*lambda)/m) * w[l]` is what causes **weight decay** — weights shrink proportionally to their magnitude.


### 🤔 How Does Regularization Prevent Overfitting?

Here are two useful mental models:
  - 💭 **Intuition 1:**
     - When `lambda` is very large, most weights shrink toward zero, simplifying the network (behaving more like basic logistic regression).
     - A well-chosen `lambda` selectively reduces only the weights responsible for overfitting.
  - 💭 **Intuition 2** (considering _tanh_ activations):
     - A very large `lambda` pushes weights close to zero, restricting operation to the near-linear region of _tanh_. The network transitions from non-linear to _approximately_ linear, becoming a _roughly_ linear classifier.
     - An appropriately tuned `lambda` linearizes just enough _tanh_ activations to curb overfitting without killing expressiveness.
     

_**Practical debugging tip**_: when implementing gradient descent, plot the cost function J against iteration count — it should decrease **monotonically**. If you're using regularization, make sure to plot the regularized J; plotting the original unregularized version may not show monotonic decrease.


### 🎯 The Dropout Technique

- Andrew Ng mentions that L2 regularization is his typical go-to.
- Dropout regularization works by randomly deactivating neurons/connections during each training iteration based on a probability threshold.
- The most widely-used implementation is called "Inverted dropout."
- Implementation code:

  ```python
  keep_prob = 0.8   # 0 <= keep_prob <= 1
  l = 3  # this code is only for layer 3
  # the generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
  d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob

  a3 = np.multiply(a3,d3)   # keep only the values in d3

  # increase a3 to not reduce the expected value of output
  # (ensures that the expected value of a3 remains the same) - to solve the scaling problem
  a3 = a3 / keep_prob       
  ```
- The mask vector d[l] is shared between forward and backward passes within the same iteration, but regenerated for every new iteration or training example.
- Dropout is disabled at test time. Applying it during inference would inject unwanted noise into predictions.

### 🧠 Dropout Intuition Deep Dive

- The core idea: dropout randomly removes units from your network each iteration, effectively training a smaller sub-network — which naturally has a regularizing effect.
- Alternative perspective: since no single feature can be relied upon, the network must distribute its learned representations across many weights.
- Mathematically, dropout produces effects similar to L2 regularization.
- You can assign different `keep_prob` values to different layers.
- The input layer's `keep_prob` should be near 1.0 (or exactly 1.0 — no dropout) to avoid discarding important features.
- If certain layers are more prone to overfitting, assign them lower `keep_prob` values. The trade-off: more hyperparameters to search. An alternative is to apply dropout selectively — only to specific layers — keeping just one `keep_prob` to tune.
- Dropout is especially prevalent in Computer Vision, where input dimensionality is enormous and labeled data is typically scarce, making overfitting a persistent concern.
- One drawback: the cost function J becomes ill-defined with dropout active, complicating debugging (e.g., plotting J vs. iterations).
  - Workaround: temporarily disable dropout (set all `keep_prob` to 1), verify J decreases monotonically, then re-enable dropout.

### 🧪 Additional Regularization Approaches

- **📸 Data Augmentation:**
  - In computer vision scenarios:
    - Horizontally flipping all images instantly doubles your effective dataset size.
    - Applying random crops, rotations, and position shifts generates additional training samples.
  - In OCR applications, you can distort and rotate digits/letters randomly.
  - Augmented data isn't as valuable as genuinely independent new samples, but it still serves as an effective regularizer.
- **⏹️ Early Stopping:**
  - Plot both training and dev set costs at each iteration. At some point, the dev cost stops improving and starts climbing.
  - Select the checkpoint where training cost is low AND dev cost is minimal.
  - Use those weights as your final model parameters.
    - ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//02-_Early_stopping.png)
  - Andrew generally favors L2 regularization over early stopping, since the latter simultaneously tries to minimize cost AND prevent overfitting — violating the orthogonalization principle (covered later).
  - However, early stopping has the advantage of not requiring an extra hyperparameter (unlike `lambda` in L2 regularization).
- **🤝 Model Ensembles:**
  - The approach:
    - Train several independent models
    - Average their predictions at inference time
  - This can yield an extra ~2% performance boost.
  - Ensembling reduces generalization error.
  - You can even ensemble snapshots of the same model captured at different points during training.

### 📏 Standardizing Your Inputs

- Normalizing inputs can dramatically speed up the training process.
- The normalization procedure:
  1. Calculate the training set mean: `mean = (1/m) * sum(x(i))`
  2. Center the data by subtracting the mean: `X = X - mean`
     - This shifts your inputs to be centered around 0.
  3. Compute the training set variance: `variance = (1/m) * sum(x(i)^2)`
  4. Scale by the variance: `X /= variance`
- Apply these same transformations (using the training set's mean and variance) to dev and test data.
- Why bother normalizing?
  - Without normalization, the cost landscape becomes elongated and asymmetric, causing slow optimization.
  - With normalization, the cost surface becomes more symmetric (circular in 2D), enabling the use of larger learning rates and much faster convergence.

### 📉 The Vanishing & Exploding Gradient Problem

- Gradients can become extremely small (vanishing) or extremely large (exploding) during training.
- To illustrate, imagine a deep network with L layers, all using **linear** activations and `b = 0` for every layer:
  - The output becomes:   
    ```
    Y' = W[L]W[L-1].....W[2]W[1]X
    ```
  - With 2 hidden units per layer and x1 = x2 = 1:

    ```
    if W[l] = [1.5   0] 
              [0   1.5] (l != L because of different dimensions in the output layer)
    Y' = W[L] [1.5  0]^(L-1) X = 1.5^L 	# which will be very large
              [0  1.5]
    ```
    ```
    if W[l] = [0.5  0]
              [0  0.5]
    Y' = W[L] [0.5  0]^(L-1) X = 0.5^L 	# which will be very small
              [0  0.5]
    ```
- This demonstrates that activations (and similarly, gradients) grow or shrink exponentially with the number of layers.
- When W > I (Identity matrix), activations and gradients explode.
- When W < I (Identity matrix), activations and gradients vanish.
- Microsoft trained a 152-layer network (ResNet)! With that much depth, exponential scaling of activations or gradients becomes a serious impediment. Tiny gradients cause gradient descent to crawl, barely learning anything.
- A partial remedy — not a complete fix — is thoughtful weight initialization (covered next).

### 🎯 Smart Weight Initialization Strategies

- Careful random initialization of weights offers a partial answer to the vanishing/exploding gradient issue.
- For a single neuron (perceptron): `Z = w1x1 + w2x2 + ... + wnxn`
  
  - When `n_x` is large, we want smaller initial `W` values to prevent the cost from blowing up.
- The target is to set the variance of weights to `1/n_x`.
- For networks using `tanh` activation, initialize like this:   
  ```
  np.random.rand(shape) * np.sqrt(1/n[l-1])
  ```
  or the Bengio et al. variant:   
  ```
  np.random.rand(shape) * np.sqrt(2/(n[l-1] + n[l]))
  ```
- For `ReLU` activations, using `2/n[l-1]` inside the square root works better:   
  ```
  np.random.rand(shape) * np.sqrt(2/n[l-1])
  ```
- The numerator (1 or 2) can itself be treated as a tunable hyperparameter, though it's not the first thing to optimize.
- This approach — ReLU combined with variance-aware initialization — is one of the most effective partial solutions to vanishing/exploding gradients.
- This technique is known as "He Initialization" (or "Xavier Initialization") and was published in a 2015 paper.

### 🔎 Approximating Gradients Numerically

- Gradient checking is a verification method that confirms your backpropagation implementation is correct.
- You can compute derivatives numerically using this approach:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//03-_Numerical_approximation_of_gradients.png)
- This approximation is extremely useful for catching backpropagation bugs, but it's computationally expensive (use it strictly for debugging, not during regular training).
- The implementation is straightforward.
- Gradient checking procedure:
  - First, concatenate `W[1],b[1],...,W[L],b[L]` into one large vector (`theta`)
  - Express the cost function as `L(theta)`
  - Similarly, flatten `dW[1],db[1],...,dW[L],db[L]` into a single vector (`d_theta`)
  - **The algorithm:**   
    ```
    eps = 10^-7   # small number
    for i in len(theta):
      d_theta_approx[i] = (J(theta1,...,theta[i] + eps) -  J(theta1,...,theta[i] - eps)) / 2*eps
    ```
  - Evaluate using: `(||d_theta_approx - d_theta||) / (||d_theta_approx||+||d_theta||)` (`||` denotes Euclidean norm) and check (with eps = 10^-7):
    - Below 10^-7 — excellent, backpropagation is almost certainly correct
    - Around 10^-5 — possibly OK, but examine individual components of `d_theta_approx - d_theta` for anomalies
    - At or above 10^-3 — likely a bug in your backpropagation code

### ⚠️ Practical Gradient Checking Tips

- Never run gradient checking during actual training — it's far too slow.
- Reserve gradient checking exclusively for debugging purposes.
- If the check fails, inspect individual components to pinpoint the source of the bug.
- When using L1 or L2 regularization, remember to include `lamda/(2m) * sum(W[l])` in J.
- Gradient checking is incompatible with dropout since J becomes inconsistent.
  - Disable dropout first (set `keep_prob = 1.0`), run the gradient check, then re-enable dropout.
- Execute gradient checks both at random initialization and after some training. Some bugs only manifest when w and b grow larger and aren't detectable at the very first (near-zero) iteration.

### 📝 Initialization Recap

- Weights $W^{[l]}$ must be initialized randomly to break symmetry.

- Biases $b^{[l]}$ can safely start at zero — symmetry is still broken as long as $W^{[l]}$ is random.

- Different initialization strategies produce different training outcomes.

- Random initialization ensures distinct hidden units learn different features.

- Avoid excessively large initial values.

- He initialization pairs well with networks using ReLU activations. 

### 📝 Regularization Recap

#### 1. L2 Regularization   
**Key takeaways:**   
  - The λ value is a hyperparameter best tuned using a dev set.
  - L2 regularization produces smoother decision boundaries. Excessive λ can cause "oversmoothing," introducing high bias.

**What L2 regularization does under the hood:**   
  - It operates on the premise that models with smaller weights are simpler. By adding a penalty on squared weight values to the cost function, all weights are driven toward smaller magnitudes. Large weights become too expensive. The result is a smoother model whose output varies more gradually with changes in input.

**Essential points to remember:**   
Impacts of L2 regularization on:
  - Cost computation:
    - An additional regularization term gets added to the cost
  - Backpropagation:
    - Extra gradient terms appear for weight matrices
  - Weight values:
    - Weights become smaller ("weight decay") — they're pushed toward zero
     
#### 2. Dropout   
**Key points about dropout:**   
- Dropout is a regularization mechanism.
- It's applied only during training — never at test time (don't randomly eliminate nodes during inference).
- Implement dropout in both forward and backward propagation passes.
- During training, divide each dropout layer's output by keep_prob to maintain consistent expected activation values. Example: with `keep_prob` = 0.5, roughly half the nodes are deactivated, so output is scaled down by 0.5. Dividing by 0.5 (equivalent to multiplying by 2) restores the expected output magnitude. This scaling works correctly for any keep_prob value.


## 🚀 Advanced Optimization Methods

### 📦 Mini-Batch Gradient Descent Explained

- Training neural networks on massive datasets is painfully slow. Finding a faster optimization approach is highly worthwhile.
- Consider `m = 50 million` training examples. Processing all of them in a single gradient step requires enormous computation.
  
  - 50 million examples won't fit in memory simultaneously, necessitating a different approach.
- The key insight: you can start making gradient updates before processing the entire dataset.
- Imagine partitioning m examples into **mini batches** of size 1000:
  - `X{1} = 0    ...  1000`
  - `X{2} = 1001 ...  2000`
  - `...`
  - `X{bs} = ...`
- Apply the same partitioning to both `X` and `Y`.
- Formal mini-batch definition ==> `t: X{t}, Y{t}`
- **Batch gradient descent** processes the entire dataset per update.
- **Mini-batch gradient descent** processes one mini-batch per update.
- Mini-batch pseudocode:
  ```
  for t = 1:No_of_batches                         # this is called an epoch
  	AL, caches = forward_prop(X{t}, Y{t})
  	cost = compute_cost(AL, Y{t})
  	grads = backward_prop(AL, caches)
  	update_parameters(grads)
  ```
- Everything inside an epoch should be vectorized for performance.
- Mini-batch gradient descent delivers massive speedups on large datasets.

### 📊 Deeper Look at Mini-Batch Behavior

- Unlike full-batch gradient descent where cost drops smoothly each iteration, mini-batch training produces a noisy cost curve — some ups and downs — but the overall trajectory should trend downward.
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//04-_batch_vs_mini_batch_cost.png)
- How mini-batch size affects behavior:
  - `mini batch size = m` ==> Full batch gradient descent
  - `mini batch size = 1` ==> Stochastic gradient descent (SGD)
  - `mini batch size between 1 and m` ==> Mini-batch gradient descent
- Full batch drawback:
  - Each iteration (epoch) takes extremely long
- SGD drawbacks:
  - Cost minimization is very noisy (can be mitigated with smaller learning rates)
  - Never truly converges to the exact minimum
  - Loses the speed benefits of vectorization
- Mini-batch sweet spot:
  1. Faster learning because:
      - You retain vectorization advantages
      - You make progress without waiting for all examples to be processed
  2. May not converge exactly (oscillates near the minimum) but reducing the learning rate helps
- Practical guidelines for choosing mini-batch size:
  1. For small datasets (< 2000 examples) — just use full batch gradient descent.
  2. Choose powers of 2 (aligns with memory architecture for optimal performance):
      `64, 128, 256, 512, 1024, ...`
  3. Verify that each mini-batch fits within CPU/GPU memory.
- Mini-batch size is a `hyperparameter`.

### 📈 Exponentially Weighted Moving Averages

- Before diving into optimizers that surpass basic gradient descent, you need to understand exponentially weighted averages.
- Imagine daily temperature readings throughout the year:
  ```
  t(1) = 40
  t(2) = 49
  t(3) = 45
  ...
  t(180) = 60
  ...
  ```
- Temperatures run low in winter and high in summer. A raw plot looks noisy.
- Computing the exponentially weighted average:
  ```
  V0 = 0
  V1 = 0.9 * V0 + 0.1 * t(1) = 4		# 0.9 and 0.1 are hyperparameters
  V2 = 0.9 * V1 + 0.1 * t(2) = 8.5
  V3 = 0.9 * V2 + 0.1 * t(3) = 12.15
  ...
  ```
- The general formula:
  ```
  V(t) = beta * v(t-1) + (1-beta) * theta(t)
  ```
- Plotting this smoothed series approximates an average over `~ (1 / (1 - beta))` data points:
    - `beta = 0.9` averages roughly the last 10 values
    - `beta = 0.98` averages roughly the last 50 values
    - `beta = 0.5` averages roughly the last 2 values
- The optimal beta for most scenarios falls between 0.9 and 0.98.
- A real-world visualization:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//Nasdaq1_small.png)   
    _(sourced from [investopedia.com](https://www.investopedia.com/))_

### 🔍 Intuition Behind Weighted Averages

- Building intuition:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//05-_exponentially_weighted_averages_intuitions.png)
- A sliding window approach would give more precise results, but the exponentially weighted average algorithm is significantly more memory-efficient and faster in practice.
- The algorithm itself is elegantly simple:
  ```
  v = 0
  Repeat
  {
  	Get theta(t)
  	v = beta * v + (1-beta) * theta(t)
  }
  ```

### 🔧 Correcting Bias in Weighted Averages

- Bias correction improves the accuracy of exponentially weighted averages, particularly during early iterations.
- Since `v(0) = 0`, initial estimates are skewed — the weighted average starts too low.
- The correction formula:
  ```
  v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)
  ```
- As t grows, `(1 - beta^t)` approaches `1`, and the correction becomes negligible.

### 🏎️ Momentum-Enhanced Gradient Descent

- Gradient descent with momentum nearly always outperforms vanilla gradient descent.
- The concept: compute exponentially weighted averages of your gradients and use those smoothed values for weight updates.
- Pseudocode:
  ```
  vdW = 0, vdb = 0
  on iteration t:
  	# can be mini-batch or batch gradient descent
  	compute dw, db on current mini-batch                
  			
  	vdW = beta * vdW + (1 - beta) * dW
  	vdb = beta * vdb + (1 - beta) * db
  	W = W - learning_rate * vdW
  	b = b - learning_rate * vdb
  ```
- Momentum enables the cost function to reach its minimum more quickly and with less oscillation.
- `beta` is a `hyperparameter`. The default `beta = 0.9` works excellently in the vast majority of cases.
- In practice, most people skip the bias correction step for momentum.

### 📐 RMSprop Optimizer

- Stands for **Root Mean Square Propagation**.
- This method accelerates gradient descent convergence.
- Pseudocode:
  ```
  sdW = 0, sdb = 0
  on iteration t:
  	# can be mini-batch or batch gradient descent
  	compute dw, db on current mini-batch
  	
  	sdW = (beta * sdW) + (1 - beta) * dW^2  # squaring is element-wise
  	sdb = (beta * sdb) + (1 - beta) * db^2  # squaring is element-wise
  	W = W - learning_rate * dW / sqrt(sdW)
  	b = B - learning_rate * db / sqrt(sdb)
  ```
- RMSprop dampens movement in high-variance directions while accelerating it in low-variance directions:
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//06-_RMSprop.png)
- To prevent division by zero, add a tiny constant `epsilon` (e.g., `epsilon = 10^-8`):   
   `W = W - learning_rate * dW / (sqrt(sdW) + epsilon)`
- RMSprop often permits using larger learning rates.
- Created by Geoffrey Hinton, first presented in a [Coursera.org](https://www.coursera.org/) lecture.

### 🏆 The Adam Optimizer

- Stands for **Adaptive Moment Estimation**.
- Adam is one of the most reliable optimizers across diverse neural network architectures, alongside RMSprop.
- It elegantly merges momentum and RMSprop into a single algorithm.
- Pseudocode:
  ```
  vdW = 0, vdW = 0
  sdW = 0, sdb = 0
  on iteration t:
  	# can be mini-batch or batch gradient descent
  	compute dw, db on current mini-batch                
  			
  	vdW = (beta1 * vdW) + (1 - beta1) * dW     # momentum
  	vdb = (beta1 * vdb) + (1 - beta1) * db     # momentum
  			
  	sdW = (beta2 * sdW) + (1 - beta2) * dW^2   # RMSprop
  	sdb = (beta2 * sdb) + (1 - beta2) * db^2   # RMSprop
  			
  	vdW = vdW / (1 - beta1^t)      # fixing bias
  	vdb = vdb / (1 - beta1^t)      # fixing bias
  			
  	sdW = sdW / (1 - beta2^t)      # fixing bias
  	sdb = sdb / (1 - beta2^t)      # fixing bias
  					
  	W = W - learning_rate * vdW / (sqrt(sdW) + epsilon)
  	b = B - learning_rate * vdb / (sqrt(sdb) + epsilon)
  ```
- Adam hyperparameters:
  - **Learning rate:** requires tuning
  - `beta1`: momentum parameter — default recommendation is `0.9`
  - `beta2`: RMSprop parameter — default recommendation is `0.999`
  - `epsilon`: default recommendation is `10^-8`

### 📉 Scheduling Learning Rate Decay

- Gradually reducing the learning rate over time can improve convergence.
- Recall that mini-batch gradient descent doesn't converge to the exact optimum. By decaying the learning rate, the step sizes shrink near the minimum, getting much closer to it.
- A common decay formula: `learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0`  
  - `epoch_num` counts passes over the full dataset, not individual mini-batches.
- Alternative continuous decay schedules:
  - `learning_rate = (0.95 ^ epoch_num) * learning_rate_0`
  - `learning_rate = (k / sqrt(epoch_num)) * learning_rate_0`
- Some practitioners apply discrete learning rate drops — reducing the rate after predetermined epoch intervals.
- Others manually adjust the learning rate during training.
- `decay_rate` introduces yet another `hyperparameter`.
- In Andrew Ng's hierarchy of priorities, learning rate decay ranks relatively low.

### 🏔️ Local Optima in High Dimensions

- In high-dimensional spaces, true local optima are extremely unlikely. A point would need to be a local minimum along every single dimension simultaneously — vanishingly improbable.
- Getting trapped at a bad local optimum is far less likely than encountering a saddle point, which isn't actually problematic.
- The real concern is **plateaus** — regions where the gradient stays near zero for extended stretches:
  - A plateau is a flat region of the cost landscape where the derivative is approximately zero.
  - This is precisely where momentum, RMSprop, and Adam prove their worth.



## 🔬 Hyperparameter Search, Batch Norm & Frameworks

### 🎛️ The Art of Hyperparameter Tuning

- Systematic hyperparameter tuning is essential for peak model performance.
- Andrew Ng's rough priority ranking for hyperparameters:
  1. Learning rate
  2. Momentum beta
  3. Mini-batch size
  4. Number of hidden units
  5. Number of layers
  6. Learning rate decay rate
  7. Regularization lambda
  8. Activation function choices
  9. Adam's `beta1` & `beta2`
- Which hyperparameter matters most depends heavily on your specific problem.
- One tuning strategy: define a grid of `N` hyperparameter combinations and evaluate each one.
- Better approach: sample randomly rather than using a fixed grid.
- Use a `Coarse to fine sampling scheme`:
  - Once you identify promising hyperparameter regions, zoom in and sample more densely within that neighborhood.
- These search strategies can be automated.

### 📊 Picking the Right Scale for Hyperparameters

- When a hyperparameter ranges from "a" to "b," searching on a logarithmic scale is generally superior to linear sampling:
  - Compute: `a_log = log(a)  # e.g. a = 0.0001 then a_log = -4`
  - Compute: `b_log = log(b)  # e.g. b = 1  then b_log = 0`
  - Sample:
    ```
    r = (a_log - b_log) * np.random.rand() + b_log
    # In the example the range would be from [-4, 0] because rand range [0,1)
    result = 10^r
    ```
    This generates uniform samples on a log scale across [a,b].
- Applying this to the momentum parameter "beta":
  - Beta's useful range is typically 0.9 to 0.999.
  - Instead, search for `1 - beta in range 0.001 to 0.1 (1 - 0.9 and 1 - 0.999)` using `a = 0.001` and `b = 0.1`:
    ```
    a_log = -3
    b_log = -1
    r = (a_log - b_log) * np.random.rand() + b_log
    beta = 1 - 10^r   # because 1 - beta = 10^r
    ```

### 🐼 Tuning Strategies: Babysitting vs. Parallel Training

- Hyperparameter intuitions from one domain may or may not transfer to a different application.
- When computational resources are limited, try the **"babysitting" (Panda) approach**:
  - Start with random initialization and begin training on Day 0.
  - Monitor the learning curve as cost gradually decreases.
  - Each day, manually tweak parameters slightly based on observed behavior.
- When you have ample compute, use the **"Caviar" approach**:
  - Launch multiple models in parallel with different hyperparameter settings.
  - At the end, compare results and pick the best performer.

### 📐 Batch Normalization of Hidden Activations

- Among the most transformative innovations in modern deep learning is **batch normalization**, introduced by Sergey Ioffe and Christian Szegedy.
- Batch normalization accelerates the training process significantly.
- We previously saw how normalizing inputs (subtracting mean, dividing by variance) reshapes the cost landscape for faster optimization.
- The natural question: *can we normalize hidden layer activations `A[l]` to speed up learning of `W[l]` and `b[l]`?* That's exactly what batch normalization does.
- There's an ongoing discussion about whether to normalize pre-activation values `Z[l]` or post-activation values `A[l]`. In practice, normalizing `Z[l]` is far more common, and that's what Andrew Ng teaches.
- The algorithm:
  - Given `Z[l] = [z(1), ..., z(m)]`, i = 1 to m (for each input)
  - Compute `mean = 1/m * sum(z[i])`
  - Compute `variance = 1/m * sum((z[i] - mean)^2)`
  - Then `Z_norm[i] = (z(i) - mean) / np.sqrt(variance + epsilon)` (add `epsilon` for numerical stability if variance = 0)
    - This forces inputs to have zero mean and unit variance.
  - Then `Z_tilde[i] = gamma * Z_norm[i] + beta`
    - This allows the network to learn a different target distribution (custom mean and variance).
    - gamma and beta are trainable model parameters.
    - The network learns the optimal output distribution.
    - _Note:_ if `gamma = sqrt(variance + epsilon)` and `beta = mean` then `Z_tilde[i] = Z_norm[i]`

### 🔗 Integrating Batch Norm into Your Network

- Batch norm applied in a 3-hidden-layer network:
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//bn.png)
- The full parameter set becomes:
  - `W[1]`, `b[1]`, ..., `W[L]`, `b[L]`, `beta[1]`, `gamma[1]`, ..., `beta[L]`, `gamma[L]`
  - `beta[1]`, `gamma[1]`, ..., `beta[L]`, `gamma[L]` are optimized using any standard algorithm (GD, RMSprop, Adam)
- Most deep learning frameworks provide built-in batch norm implementations:
  
  - Example in TensorFlow: `tf.nn.batch-normalization()`
- Batch normalization is typically used in conjunction with mini-batches.
- An important subtlety: when using batch norm, the bias parameters `b[1]`, ..., `b[L]` become redundant because the mean subtraction step cancels them out:
  ```
  Z[l] = W[l]A[l-1] + b[l] => Z[l] = W[l]A[l-1]
  Z_norm[l] = ...
  Z_tilde[l] = gamma[l] * Z_norm[l] + beta[l]
  ```
  - Computing the mean of a constant `b[l]` and subtracting it effectively eliminates `b[l]`.
- When using batch normalization, you can either remove b[l] entirely or fix it at zero.
- The effective parameters per layer become `W[l]`, `beta[l]`, and `alpha[l]`.
- Tensor shapes:
  - `Z[l]       - (n[l], m)`
  - `beta[l]    - (n[l], m)`
  - `gamma[l]   - (n[l], m)`

### 💡 Why Batch Normalization is So Effective

- The first reason mirrors why we normalize the input features X.
- The second reason: batch norm mitigates the problem of shifting input distributions between layers (internal covariate shift).
- Batch normalization also provides a mild regularization effect:
  - Each mini-batch is normalized using its own mean/variance statistics.
  - This injects some noise into the `Z[l]` values, similar to how dropout adds noise to activations.
  - The result is a slight regularization effect.
  - Larger mini-batches reduce this noise and diminish the regularization effect.
  - Don't depend on batch norm for regularization — its primary purpose is normalizing hidden activations to speed up learning. Use dedicated regularization methods (L2, dropout) instead.

### 🧪 Handling Batch Norm During Inference

- During training, batch norm computes mean and variance from each mini-batch.
- At test time, you may need to process individual examples. Computing mean and variance from a single sample is meaningless.
- Solution: maintain running estimates of mean and variance computed during training.
- These running estimates are then used at inference time.
- This technique is sometimes called "Running average."
- Most deep learning frameworks handle this automatically with sensible defaults.

### 🎯 Multi-Class Classification with Softmax

- Every example so far has involved binary classification.
- Softmax regression extends logistic regression to handle multiple classes.
- Example with 4 categories: `dog`, `cat`, `baby chick`, and `none`:
  - Dog `class = 1`
  - Cat `class = 2`
  - Baby chick `class = 3`
  - None `class = 0`
  - Dog vector: `y = [0 1 0 0]`
  - Cat vector: `y = [0 0 1 0]`
  - Baby chick vector: `y = [0 0 0 1]`
  - None vector: `y = [1 0 0 0]`
- Key notation:
  - `C = no. of classes`
  - Class range: `(0, ..., C-1)`
  - Output layer size: `Ny = C`
- Each of the C output neurons produces a probability for its respective class.
- The final layer uses the Softmax activation function rather than sigmoid.
- Softmax equations:
  ```
  t = e^(Z[L])                      # shape(C, m)
  A[L] = e^(Z[L]) / sum(t)          # shape(C, m), sum(t) - sum of t's for each example (shape (1, m))
  ```

### 🎓 Training the Softmax Classifier

- Hard max assigns 1 to the highest value and 0 to everything else.
  
  - In NumPy, this is `np.max` along the vertical axis.
- "Softmax" gets its name from producing smooth (soft) probability distributions rather than hard binary decisions.
- Softmax generalizes the logistic activation to `C` classes. When `C = 2`, softmax reduces to standard logistic regression.
- The loss function for softmax:
  ```
  L(y, y_hat) = - sum(y[j] * log(y_hat[j])) # j = 0 to C-1
  ```
- The cost function for softmax:
  ```
  J(w[1], b[1], ...) = - 1 / m * (sum(L(y[i], y_hat[i]))) # i = 0 to m
  ```
- Backpropagation gradient for softmax:
  ```
  dZ[L] = Y_hat - Y
  ```
- The softmax derivative:
  ```
  Y_hat * (1 - Y_hat)
  ```
- Illustrated example:
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/2-%20Improving%20Deep%20Neural%20Networks/Images//07-_softmax.png)

### 🛠️ Choosing a Deep Learning Framework

- Implementing everything from scratch isn't practical. Our NumPy code served to build understanding of how NNs work internally.
- Numerous excellent deep learning frameworks exist today.
- The field has matured to the point where using frameworks (rather than raw implementations) is the norm for productive work.
- Leading frameworks include:
  - Caffe / Caffe2
  - CNTK
  - DL4j
  - Keras
  - Lasagne
  - mxnet
  - PaddlePaddle
  - TensorFlow
  - Theano
  - Torch / PyTorch
- These frameworks improve rapidly. A comparative overview is available [here](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software).
- Criteria for selecting a framework:
  - Development and deployment ease
  - Execution speed
  - Genuinely open source with transparent governance
- A good framework doesn't just save coding time — it can also apply optimizations that make your code run faster.

### 🔷 Getting Started with TensorFlow

- This section introduces the fundamental structure of TensorFlow programs.
- Let's implement a simple minimization:
  - Target function: `J(w) = w^2 - 10w + 25`
  - Expected result: `w = 5` since the function equals `(w-5)^2 = 0`
  - Version 1:
    ```python
    import numpy as np
    import tensorflow as tf
    
    
    w = tf.Variable(0, dtype=tf.float32)                 # creating a variable w
    cost = tf.add(tf.add(w**2, tf.multiply(-10.0, w)), 25.0)        # can be written as this - cost = w**2 - 10*w + 25
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    session.run(w)    # Runs the definition of w, if you print this it will print zero
    session.run(train)

    print("W after one iteration:", session.run(w))

    for i in range(1000):
    	session.run(train)

    print("W after 1000 iterations:", session.run(w))
    ```
  - Version 2 (feeding inputs as coefficients):

    ```python
    import numpy as np
    import tensorflow as tf
    
    
    coefficients = np.array([[1.], [-10.], [25.]])

    x = tf.placeholder(tf.float32, [3, 1])
    w = tf.Variable(0, dtype=tf.float32)                 # Creating a variable w
    cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    session.run(w)    # Runs the definition of w, if you print this it will print zero
    session.run(train, feed_dict={x: coefficients})

    print("W after one iteration:", session.run(w))

    for i in range(1000):
    	session.run(train, feed_dict={x: coefficients})

    print("W after 1000 iterations:", session.run(w))
    ```
- TensorFlow handles backpropagation automatically — you only need to define the forward pass.
- A placeholder in TensorFlow is a variable whose value gets assigned later.
- For mini-batch training, swap `feed_dict={x: coefficients}` with the current mini-batch data each iteration.
- A widely-used TensorFlow pattern:
  ```python
  with tf.Session() as session:       # better for cleaning up in case of error/exception
  	session.run(init)
  	session.run(w)
  ```
- Deep learning frameworks let you accomplish complex operations — such as switching optimizers — with a single line of code.
_**Additional notes:**_
- The standard TensorFlow workflow consists of:
  1. Declare Tensors (variables) without immediately executing them.
  2. Define operations connecting those Tensors.
  3. Initialize your Tensors.
  4. Create a Session.
  5. Run the Session to execute the operations defined above.
- Instead of manually coding the cost function, TensorFlow offers:
  `tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)`
- To initialize neural network weights in TensorFlow:
  ```
  W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

  b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
  ```
- In a 3-layer network, forward propagation stops at `Z3`. TensorFlow's loss computation functions expect the final linear layer output directly, so computing `A3` is unnecessary.
- To clear the computational graph: `tf.reset_default_graph()`

## 📓 Supplementary Notes

- For cutting-edge deep learning research, check the ICLR proceedings (or NIPS proceedings) — they offer an excellent snapshot of the field's current state.
- About Yuanqing Lin:
  - Led Baidu's research division
  - First to achieve an ImageNet victory
  - Contributed to the PaddlePaddle deep learning framework
