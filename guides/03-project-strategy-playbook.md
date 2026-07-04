
# 🎯 Organizing and Directing Machine Learning Initiatives

This guide covers the third module in the deep learning track on [Coursera](https://www.coursera.org/specializations/deep-learning), facilitated by [DeepLearning.ai](http://deeplearning.ai/). Instruction is provided by Andrew Ng.

# 📚 Complete Guide Collection

* [**Module 1: Building Blocks of Neural Networks & Deep Learning**](guides/01-foundations-of-neural-nets.md)
* [**Module 2: Enhancing Deep Networks — Hyperparameter Tuning, Regularization & Optimization**](guides/02-tuning-and-optimization.md)
* [**Module 3: Organizing and Directing Machine Learning Initiatives**](guides/03-project-strategy-playbook.md)
* [**Module 4: Image Processing with Convolutional Networks**](guides/04-visual-recognition-networks.md)
* [**Module 5: Sequential Data and Temporal Models**](guides/05-temporal-sequence-learning.md)

## 📑 Navigation

* [Organizing and Directing Machine Learning Initiatives](#organizing-and-directing-machine-learning-initiatives)
   * [Navigation](#navigation)
   * [What This Module Covers](#what-this-module-covers)
   * [Strategic Approaches — Part 1](#strategic-approaches--part-1)
      * [The Need for ML Strategy](#the-need-for-ml-strategy)
      * [Independent Tuning Controls](#independent-tuning-controls)
      * [Unified Performance Score](#unified-performance-score)
      * [Threshold vs. Optimization Metrics](#threshold-vs-optimization-metrics)
      * [Splitting Data into Train/Dev/Test](#splitting-data-into-traindevtest)
      * [Choosing Dev and Test Set Sizes](#choosing-dev-and-test-set-sizes)
      * [Revising Your Metrics and Data Splits](#revising-your-metrics-and-data-splits)
      * [Benchmarking Against Humans](#benchmarking-against-humans)
      * [Reducible Bias](#reducible-bias)
      * [Interpreting Human Baselines](#interpreting-human-baselines)
      * [Going Beyond Human Accuracy](#going-beyond-human-accuracy)
      * [Boosting Model Quality](#boosting-model-quality)
   * [Strategic Approaches — Part 2](#strategic-approaches--part-2)
      * [Systematic Error Investigation](#systematic-error-investigation)
      * [Fixing Mislabeled Examples](#fixing-mislabeled-examples)
      * [Ship Fast and Refine Iteratively](#ship-fast-and-refine-iteratively)
      * [Working with Mismatched Distributions](#working-with-mismatched-distributions)
      * [Diagnosing Bias/Variance Under Distribution Shift](#diagnosing-biasvariance-under-distribution-shift)
      * [Tackling Data Mismatch](#tackling-data-mismatch)
      * [Leveraging Pre-trained Knowledge](#leveraging-pre-trained-knowledge)
      * [Joint Multi-objective Training](#joint-multi-objective-training)
      * [Understanding End-to-End Deep Learning](#understanding-end-to-end-deep-learning)
      * [Deciding on End-to-End Approaches](#deciding-on-end-to-end-approaches)

## 🔍 What This Module Covers

Below is the official description from the course [page](https://www.coursera.org/learn/machine-learning-projects):

> You will learn how to build a successful machine learning project. If you aspire to be a technical leader in AI, and know how to set direction for your team's work, this course will show you how.
>
> Much of this content has never been taught elsewhere, and is drawn from my experience building and shipping many deep learning products. This course also has two "flight simulators" that let you practice decision-making as a machine learning project leader. This provides "industry experience" that you might otherwise get only after years of ML work experience.
>
> After 2 weeks, you will: 
> - Understand how to diagnose errors in a machine learning system, and 
> - Be able to prioritize the most promising directions for reducing error
> - Understand complex ML settings, such as mismatched training/test sets, and comparing to and/or surpassing human-level performance
> - Know how to apply end-to-end learning, transfer learning, and multi-task learning
>
> I've seen teams waste months or years through not understanding the principles taught in this course. I hope this two week course will save you months of time.
>
> This is a standalone course, and you can take this so long as you have basic machine learning knowledge. This is the third course in the Deep Learning Specialization.



## 🧭 Strategic Approaches — Part 1

### 🤔 The Need for ML Strategy

- When trying to boost accuracy in your deep learning pipeline, you face a long list of possible next steps:
  - Gather additional training samples.
  - Diversify the training data distribution.
  - Run gradient descent for more epochs.
  - Experiment with alternative optimizers (e.g., Adam).
  - Scale up the network architecture.
  - Scale down the network architecture.
  - Introduce dropout layers.
  - Apply L2 weight penalties.
  - Modify the architecture itself (swap activation functions, adjust hidden unit counts, etc.)
- This module equips you with frameworks to analyze your situation and channel effort toward the most impactful improvements.

### 🎛️ Independent Tuning Controls

- Skilled deep learning practitioners intuitively know which knob to turn to achieve a particular effect. We refer to this principle as orthogonalization.
- With orthogonalized controls, every adjustment targets a single aspect of performance without interfering with others.
- For a supervised learning system to work effectively, you need to satisfy a chain of requirements in order:
  1. Achieve strong training set performance relative to your cost function (ideally approaching human-level accuracy).
     - If this fails, consider a larger network or switching optimization algorithms (e.g., Adam)...
  2. Achieve solid dev set performance on the cost function.
     - If this fails, consider regularization techniques or expanding the training set...
  3. Achieve reliable test set performance on the cost function.
     - If this fails, consider enlarging the dev set...
  4. Deliver real-world results.
     - If this fails, consider redesigning the dev set or revising the cost function...

### 📊 Unified Performance Score

- It's much more efficient to define a single numerical evaluation metric upfront before launching your project.
- Understanding the distinction between precision and recall (using a cat classifier as an illustration):
  - Imagine running the classifier on 10 https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/3-%20Structuring%20Machine%20Learning%20Projects/Images/ containing 5 cats and 5 non-cats. The model flags 4 images as cats but misclassifies 1 non-cat.
  - The confusion matrix looks like:

      |                | Predicted cat  | Predicted non-cat |
      | -------------- | -------------- | ----------------- |
      | Actual cat     | 3              | 2                 |
      | Actual non-cat | 1              | 4                 |
  - **Precision**: fraction of correctly identified cats among all predicted cats: P = 3/(3 + 1) 
  - **Recall**: fraction of actual cats that the model successfully identified: R = 3/(3 + 2)
  - **Accuracy**: (3+4)/10
- Relying on precision and recall individually for evaluation works in many scenarios, but taken separately they don't reveal which algorithm wins. For instance:

  | Classifier | Precision | Recall |
  | ---------- | --------- | ------ |
  | A          | 95%       | 90%    |
  | B          | 98%       | 85%    |
- A smarter approach is to merge precision and recall into one unified metric. The `F1` score does exactly this:
  - Think of `F1` as the harmonic mean of precision and recall:
    `F1 = 2 / ((1/P) + (1/R))`

### ⚖️ Threshold vs. Optimization Metrics

- Sometimes a single number metric isn't straightforward to define. Consider:

  | Classifier | F1   | Running time |
  | ---------- | ---- | ------------ |
  | A          | 90%  | 80 ms        |
  | B          | 92%  | 95 ms        |
  | C          | 92%  | 1,500 ms     |
- The solution is to designate one metric as the optimization target and treat the rest as constraints. For instance:
  ```
  Maximize F1                     # optimizing metric
  subject to running time < 100ms # satisficing metric
  ```
- As a universal guideline:
  ```
  Maximize 1     # optimizing metric (one optimizing metric)
  subject to N-1 # satisficing metric (N-1 satisficing metrics)
  ```

### 📂 Splitting Data into Train/Dev/Test

- Your dev and test partitions must be drawn from an identical distribution.
- Select dev and test sets that mirror the real-world data you expect to encounter and care most about handling well.
- Defining the dev set along with your evaluation metric is essentially choosing the bullseye you're aiming for.

### 📏 Choosing Dev and Test Set Sizes

- Traditionally, data was split 70% train / 30% test or 60% train / 20% dev / 20% test.
- That legacy approach was reasonable when working with fewer than ~100,000 samples.
- In contemporary deep learning with a million or more examples, a sensible partition might be 98% train, 1% dev, 1% test.

### 🔄 Revising Your Metrics and Data Splits

- Consider this scenario: in a cat classification task, we observe these outcomes:

  | Metric      | Classification error                                         |
  | ----------- | ------------------------------------------------------------ |
  | Algorithm A | 3% error (But a lot of porn https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/3-%20Structuring%20Machine%20Learning%20Projects/Images/ are treated as cat https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/3-%20Structuring%20Machine%20Learning%20Projects/Images/ here) |
  | Algorithm B | 5% error                                                     |
  - Judging purely by the metric, "A" wins — but from a user's perspective, "B" is clearly preferable.
  - This signals that our metric needs to be redesigned.
  - `OldMetric = (1/m) * sum(y_pred[i] != y[i] ,m)`
    - Here m represents the total dev set examples.
  - `NewMetric = (1/sum(w[i])) * sum(w[i] * (y_pred[i] != y[i]) ,m)`
    - where:
       - `w[i] = 1                   if x[i] is not porn`
       - `w[i] = 10                 if x[i] is porn`

- This illustrates orthogonalization applied to problem decomposition:

  1. Determine a metric that faithfully captures your objective — set the target.
  2. Focus on achieving strong performance against that metric — aim precisely at the target.

- Key takeaway: if excelling on your metric + dev/test set doesn't translate to excelling in your real application, then revise your metric and/or your dev/test set.

### 👤 Benchmarking Against Humans

- We compare against human-level accuracy for two primary reasons:
  1. Deep learning has advanced to the point where ML systems can realistically compete with humans across many domains.
  2. The design and development workflow becomes significantly more productive when targeting a task that humans can also perform.
- Once an algorithm approaches human-level capability, subsequent gains in accuracy tend to plateau.
    ![01- Why human-level performance](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/3-%20Structuring%20Machine%20Learning%20Projects/Images//01-_Why_human-level_performance.png)
- No system can beat the theoretical floor known as "Bayes optimal error."
- The gap between human-level error and Bayes optimal error is typically very narrow.
- Since humans excel at many tasks, as long as your ML system trails human performance, you can:
  - Collect human-annotated labels.
  - Perform manual error analysis to understand why humans succeed.
  - Conduct sharper bias/variance diagnostics.

### 📉 Reducible Bias

- Imagine the cat classifier yields these numbers:

  | Humans             | 1%   | 7.5% |
  | ------------------ | ---- | ---- |
  | **Training error** | 8%   | 8%   |
  | **Dev Error**      | 10%  | 10%  |
  - Left column: human error at 1% means the priority should be reducing **bias**.
  - Right column: human error at 7.5% means the priority should be reducing **variance**.
  - Human-level error serves as a stand-in (proxy) for Bayes optimal error. Bayes error is always lower (better), but human performance is usually a close approximation.
  - You can't outperform Bayes error without overfitting.
  - `Avoidable bias = Training error - Human (Bayes) error`
  - `Variance = Dev error - Training error`

### 🧠 Interpreting Human Baselines

- When selecting a human-level benchmark, the choice should align with the objectives of your system.
- There may be several human-level performance figures depending on expertise. Pick the one most appropriate as a Bayes error proxy for your specific application.
- Pushing beyond human-level accuracy becomes increasingly difficult for deep learning models.
- Framework for bias/variance analysis using human baselines:
  1. Human-level error (proxy for Bayes error)
     - Compute `avoidable bias = training error - human-level error`
     - If **avoidable bias** is the dominant gap, you're facing a *bias* issue and should adopt **bias reduction** techniques.
  2. Training error
     - Compute `variance = dev error - training error`
     - If **variance** is the dominant gap, you should adopt **variance reduction** techniques.
  3. Dev error
- Having a solid estimate of human-level performance gives you a Bayes error approximation. This enables faster decisions about whether to prioritize bias reduction or variance reduction.
- These approaches remain effective until you exceed human performance, at which point your Bayes error estimate becomes less reliable for guiding such decisions.

### 🚀 Going Beyond Human Accuracy

- Deep learning has already eclipsed human capability in certain domains, such as:
  - Online advertising click prediction.
  - Product recommendation engines.
  - Loan approval decisions.
- These examples involve structured data problems, not natural perception tasks. Humans maintain a substantial edge in perception-heavy areas like vision and speech recognition.
- Surpassing human performance on natural perception tasks is harder for machines, though some systems have managed to do so.

### 🛠️ Boosting Model Quality

- Supervised learning rests on two core assumptions:
  1. The model can fit the training data well — meaning low **avoidable bias** is achievable.
  2. Training performance transfers well to the dev/test set — meaning **variance** remains manageable.
- Follow this roadmap to enhance your deep learning supervised system:
  1. Examine the gap between human-level error and training error — this is **avoidable bias**.
  2. Examine the gap between dev/test error and training error — this is **variance**.
  3. If **avoidable bias** is substantial, consider:
     - Increasing model capacity.
     - Training longer or switching optimizers (Momentum, RMSprop, Adam).
     - Searching over different architectures and hyperparameters.
  4. If **variance** is substantial, consider:
     - Acquiring more training data.
     - Applying regularization (L2, Dropout, data augmentation).
     - Searching over different architectures and hyperparameters.



## 🔬 Strategic Approaches — Part 2

### 🔎 Systematic Error Investigation

- Error analysis involves manually inspecting mistakes produced by your algorithm to discover actionable patterns. For example:
  - In the cat classifier scenario, suppose you have 10% error on your dev set and want to bring it down.
  - You notice that some misclassified images are dog photos that resemble cats. Is it worth spending weeks making the classifier better at distinguishing dogs?
  - A structured error analysis approach:
    - Randomly select 100 mislabeled dev set examples.
    - Tally how many are actually dogs.
    - If only 5 out of 100 are dogs, fixing this would reduce error by at most 0.5% (to ~9.5%) — the ceiling is too low.
    - If 50 out of 100 are dogs, the potential improvement to ~5% makes this effort worthwhile.
- Error analysis prevents you from spending significant time on changes that would have minimal impact.
- You can assess multiple improvement ideas simultaneously by creating a tracking spreadsheet:

  | Image        | Dog    | Great Cats | blurry  | Instagram filters |    Comments    |
  | ------------ | ------ | ---------- | ------- | ----------------- |--------------- |
  | 1            | ✓      |            |         | ✓                 |  Pitbul        |
  | 2            | ✓      |            | ✓       | ✓                 |                |
  | 3            |        |            |         |                   |Rainy day at zoo|
  | 4            |        | ✓          |         |                   |                |
  | ....         |        |            |         |                   |                |
  | **% totals** | **8%** | **43%**    | **61%** |      **12%**      |                |
- From this table, you'd prioritize tackling blurry https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/3-%20Structuring%20Machine%20Learning%20Projects/Images/ or great cats to get the biggest performance lift.
- This rapid counting exercise, typically completable in just a few hours, dramatically improves your ability to prioritize and gauge the promise of different directions.

### 🏷️ Fixing Mislabeled Examples

- Deep learning models handle random labeling noise in training data reasonably well, but are more sensitive to systematic labeling errors. Still, correcting these labels when feasible is beneficial.
- To check for label errors in your dev/test set, extend your error analysis with a mislabeled column:

  | Image        | Dog    | Great Cats | blurry  | Mislabeled | Comments |
  | ------------ | ------ | ---------- | ------- | ---------- | -------- |
  | 1            | ✓      |            |         |            |          |
  | 2            | ✓      |            | ✓       |            |          |
  | 3            |        |            |         |            |          |
  | 4            |        | ✓          |         |            |          |
  | ....         |        |            |         |            |          |
  | **% totals** | **8%** | **43%**    | **61%** | **6%**     |          |
  - Breaking it down:
    - If total dev set error: 10%
      - Errors from incorrect labels: 0.6%
      - Errors from other sources: 9.4%
    - You should concentrate on the 9.4% rather than chasing the label noise.
- Keep these principles in mind when correcting dev/test labels:
  - Use the same correction process for both dev and test sets to maintain distributional consistency.
  - Audit examples the algorithm classified correctly too, not just the errors. (Though this is often skipped when accuracy is already high.)
  - Training data and dev/test data may end up with slightly different distributions after corrections.
  - Maintaining identical distributions for dev and test sets is critical. A minor distributional shift in training data is generally acceptable.

### ⚡ Ship Fast and Refine Iteratively

- The recommended workflow for launching a deep learning project:
  - Establish your dev/test set and evaluation metric.
  - Build an initial working system as quickly as possible.
  - Apply Bias/Variance and Error analysis to guide your next priorities.

### 🔀 Working with Mismatched Distributions

- Many teams building deep learning applications encounter training sets that differ in distribution from their dev/test sets, largely because deep learning is so data-hungry.
- Here are two strategies when training and dev/test distributions diverge:
  - Option A (generally not recommended): combine all data, shuffle, and randomly split into train/dev/test.
    - Upside: all partitions share the same distribution.
    - Downside: real-world data that was in your original dev/test sets becomes diluted in the new splits, which may not serve your goals.
  - Option B: incorporate some dev/test-style examples into training while keeping dev/test focused on target data.
    - Upside: your dev/test set accurately represents the distribution you care about.
    - Downside: training and dev/test distributions will differ. However, long-term performance tends to be better.

### 📐 Diagnosing Bias/Variance Under Distribution Shift

- Standard bias/variance analysis changes when training and dev/test data come from different sources.
- Example using the cat classification task. Suppose your current results are:
  - Human error: 0%
  - Train error: 1%
  - Dev error: 10%
  - This looks like a variance problem at first glance, but since the distributions differ, you can't be certain — the training set may simply have been easier.
- To resolve this ambiguity, carve out a **train-dev set** — a random subset from training data (matching its distribution). Now:
  - Human error: 0%
  - Train error: 1%
  - Train-dev error: 9%
  - Dev error: 10%
  - This confirms a genuine high variance issue.
- Consider an alternate scenario:
  - Human error: 0%
  - Train error: 1%
  - Train-dev error: 1.5%
  - Dev error: 10%
  - Here the culprit is *data mismatch* rather than variance.
- Summary of diagnostics:
  1. Human-level error (Bayes error proxy)
  2. Train error
     - Compute `avoidable bias = training error - human level error`
     - A large gap indicates a **high bias** problem — use bias reduction strategies.
  3. Train-dev error
     - Compute `variance = training-dev error - training error`
     - A large gap indicates a **high variance** problem — use variance reduction strategies.
  4. Dev error
     - Compute `data mismatch = dev error - train-dev error`
     - A gap much larger than the train-dev error indicates a **data mismatch** problem.
  5. Test error
     - Compute `degree of overfitting to dev set = test error - dev error`
     - A large positive gap suggests you may need a bigger dev set (since dev and test share a distribution, a large gap here signals overfitting to the dev set).
- Unfortunately, systematic solutions for data mismatch remain limited. The next section covers some approaches worth trying.

### 🔧 Tackling Data Mismatch

- There are no fully systematic remedies, but several approaches can help:
1. Perform manual error analysis to understand the specific differences between training and dev/test distributions.
2. Make training data more representative of dev/test data, or gather additional data that resembles the dev/test set.
- To make training data resemble dev data more closely, one technique is **artificial data synthesis**:
    - Blend existing training samples with elements that shift them toward the dev/test distribution.
      - Examples:
        1. Mix clean audio with car background noise to create noisy audio samples.
        2. Render synthetic car images using 3D graphics for a vehicle classification task.
    - Be cautious: you might inadvertently generate data from only a tiny corner of the full input space, causing the network to overfit to artifacts of the synthesis (e.g., a single noise recording or a limited set of 3D car models).

### 🔄 Leveraging Pre-trained Knowledge

- Transfer learning means taking what a model learned on task A and repurposing it for task B.
- For instance, a cat classifier trained on a large dataset can contribute its learned representations to an X-ray diagnosis problem.
- The process involves removing the final layer and its weights, then:
  1. **Small target dataset**: freeze all earlier weights, add fresh output layer(s), initialize their weights randomly, and train only these new layers on the target data.
  2. **Sufficient target data**: retrain all layers end-to-end.
- Option 1 and 2 are known as **fine-tuning**, while training on task A is called **pretraining**.
- Transfer learning is appropriate when:
  - Tasks A and B share the same input modality (e.g., images, audio).
  - You have abundant data for the source task A but limited data for the target task B.
  - Low-level representations learned in task A are relevant to task B.

### 🎯 Joint Multi-objective Training

- Unlike transfer learning's sequential approach (learn A then adapt to B), multi-task learning trains a single network on multiple objectives simultaneously, with each task potentially reinforcing the others.
- Example:
  - Building an object detector that identifies pedestrians, cars, stop signs, and traffic lights (each image can have multiple labels).
  - The label matrix Y has shape `(4, m)` since there are 4 binary classification targets.
  - The cost function is:   
  `Cost = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j))), i = 1..m, j = 1..4`, where   
  `L = - y(i)_j * log(y_hat(i)_j) - (1 - y(i)_j) * log(1 - y_hat(i)_j)`
- Training a single shared network for all four tasks often outperforms training four independent networks, because early-layer features can be reused across object types.
- Multi-task learning gracefully handles incomplete labels. For example:
  ```
  Y = [1 ? 1 ...]
      [0 0 1 ...]
      [? 1 ? ...]
  ```
  - The loss function simply skips missing entries:   
    `Loss = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j) for all j which y(i)_j != ?))`
- Multi-task learning is most effective when:
  1. The tasks share useful lower-level features.
  2. Data quantities across tasks are roughly comparable.
  3. The network is large enough to handle all tasks well.
- A sufficiently large shared network tends to outperform separate task-specific models.
- In practice, transfer learning is used more frequently than multi-task learning today.

### 🔗 Understanding End-to-End Deep Learning

- Traditional systems decompose problems into multiple pipeline stages. End-to-end deep learning replaces this entire pipeline with a single neural network.
- Example 1:
  - Speech recognition pipeline:
    ```
    Audio ---> Features --> Phonemes --> Words --> Transcript    # non-end-to-end system
    Audio ---------------------------------------> Transcript    # end-to-end deep learning system
    ```
  - The end-to-end approach grants the data more freedom — it may not rely on phonemes at all during learning!
- End-to-end systems require substantially more data than multi-stage pipelines. With limited data, the traditional staged approach may perform better.
- Example 2:
  - Face recognition pipeline:
    ```
    Image ---------------------> Face recognition    # end-to-end deep learning system
    Image --> Face detection --> Face recognition    # deep learning system - best approach for now
    ```
  - Currently, the two-stage approach works better in practice.
  - Both stages use deep learning independently.
  - This works because it's easier to obtain cropped face images for comparison than end-to-end labeled examples of people in front of cameras.
  - The second stage takes two face crops as input and determines whether they belong to the same person.
- Example 3:
  - Machine translation pipeline:
    ```
    English --> Text analysis --> ... --> French    # non-end-to-end system
    English ----------------------------> French    # end-to-end deep learning system - best approach
    ```
  - End-to-end wins here because there's sufficient parallel text data available.
- Example 4:
  - Estimating a child's age from a hand X-ray:
  ```
  Image --> Bones --> Age    # non-end-to-end system - best approach for now
  Image ------------> Age    # end-to-end system
  ```
  - The staged approach currently performs better due to insufficient data for training an end-to-end model.

### ✅ Deciding on End-to-End Approaches

- Advantages of end-to-end deep learning:
  - Lets the data speak for itself. A pure ML pipeline can capture patterns in the data without being constrained by human assumptions.
  - Reduces the need for manually engineered intermediate components.
- Disadvantages of end-to-end deep learning:
  - Often demands very large datasets.
  - Discards potentially valuable hand-designed components (which help most when data is limited).
- Guidelines for choosing end-to-end:
  - Central question: Do you have enough data to learn a mapping of the required **complexity** from x to y?
  - Consider using ML/DL for individual sub-components rather than the entire pipeline.
  - When applying supervised learning, carefully select which X→Y mappings to learn based on data availability for each sub-task.
