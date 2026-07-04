<h1 align="center">🧠 Deep Learning Specialization — Comprehensive Study Hub</h1>

<p align="center">
  <em>An organized collection of lecture notes, interactive notebooks, quizzes, and assignments from Andrew Ng's renowned deep learning curriculum on <a href="https://www.coursera.org/specializations/deep-learning">Coursera</a>, powered by <a href="http://deeplearning.ai/">DeepLearning.ai</a>.</em>
</p>

<p align="center">
  <a href="guides/01-foundations-of-neural-nets.md">📐 Foundations</a> •
  <a href="guides/02-tuning-and-optimization.md">⚡ Optimization</a> •
  <a href="guides/03-project-strategy-playbook.md">🎯 Strategy</a> •
  <a href="guides/04-visual-recognition-networks.md">🔬 Vision</a> •
  <a href="guides/05-temporal-sequence-learning.md">🔮 Sequences</a>
</p>

---

## 🗺️ Repository Map

<details open>
<summary><strong>Click to expand the full directory layout</strong></summary>

```
📂 root
├── 📁 guides/                          ← In-depth written notes per course
│   ├── 01-foundations-of-neural-nets.md
│   ├── 02-tuning-and-optimization.md
│   ├── 03-project-strategy-playbook.md
│   ├── 04-visual-recognition-networks.md
│   └── 05-temporal-sequence-learning.md
├── 📁 labs/                             ← Hands-on exercises & quizzes
│   ├── core-neural-networks/
│   ├── hyperparameter-mastery/
│   ├── ml-strategy-quizzes/
│   ├── computer-vision-deep-dive/
│   └── sequential-data-models/
├── 📁 assignments/                      ← Programming assignments
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-neural-network-basics/
│   ├── 04-support-vector-machines/
│   ├── 05-bias-variance-regularization/
│   ├── 06-clustering-and-pca/
│   ├── 07-ensemble-methods/
│   └── 08-anomaly-and-recommenders/
├── 📁 rendered-notes/                   ← Jupyter notebook versions of the guides
│   ├── neural-nets-fundamentals.ipynb
│   ├── optimization-techniques.ipynb
│   ├── ml-project-structure.ipynb
│   ├── convnets-explained.ipynb
│   └── sequence-modeling.ipynb
└── 📁 resources/
    └── comprehensive-dl-reference.pdf
```

</details>

---

## 🎯 Quick Access — Study Guides

Each course module has been distilled into a comprehensive written guide. Start here for the conceptual foundations before jumping into code.

| # | 🏗️ Module | 📐 Guide Link | 💡 Key Themes |
|:-:|-----------|:-------------|:-------------|
| 1 | **Neural Network Essentials** | [Foundations of Neural Nets](guides/01-foundations-of-neural-nets.md) | Perceptrons, forward/back-prop, activation functions, shallow & deep architectures |
| 2 | **Performance Enhancement** | [Tuning & Optimization](guides/02-tuning-and-optimization.md) | Regularization, gradient descent variants, batch norm, hyperparameter search |
| 3 | **ML Project Orchestration** | [Project Strategy Playbook](guides/03-project-strategy-playbook.md) | Train/dev/test splits, error analysis, human-level benchmarks, transfer learning |
| 4 | **Visual Recognition** | [Visual Recognition Networks](guides/04-visual-recognition-networks.md) | Convolution operations, classic CNN architectures, detection, face recognition, style transfer |
| 5 | **Temporal & Sequential Data** | [Temporal Sequence Learning](guides/05-temporal-sequence-learning.md) | RNNs, LSTMs, GRUs, word embeddings, attention mechanisms, sequence-to-sequence |

---

## 🔬 Course Breakdown

### 📐 Module 1 — Building Blocks of Neural Networks

> *Understand the engine that powers modern AI — from a single neuron to deep stacked layers.*

![](https://systweak1.vo.llnwd.net/content/wp/systweakblogsnew/uploads_new/2018/03/hidden-layers-in-network.gif)

<details>
<summary>🔍 What's covered in this module?</summary>

- [Defining and understanding neural networks](guides/01-foundations-of-neural-nets.md#what-is-a-neural-network-nn) — how information flows through layers of connected nodes
- [Core math behind neural nets](guides/01-foundations-of-neural-nets.md#neural-networks-basics) — supervised learning setups, loss functions, and gradient computation
- [Designing single-layer networks](guides/01-foundations-of-neural-nets.md#shallow-neural-networks) — shallow architectures and their capabilities
- [Scaling to depth](guides/01-foundations-of-neural-nets.md#deep-neural-networks) — why deeper networks unlock greater representational power

</details>

📓 **Rendered notebook:** [`neural-nets-fundamentals.ipynb`](rendered-notes/neural-nets-fundamentals.ipynb)

---

### ⚡ Module 2 — Making Deep Networks Work Better

> *Diagnose, tune, and supercharge your neural networks with proven optimization strategies.*

![](https://i.pinimg.com/originals/63/62/8f/63628f546ad55fd31091e23c623cb9f5.gif)

<details>
<summary>🔍 What's covered in this module?</summary>

- [Hands-on deep learning practices](guides/02-tuning-and-optimization.md#practical-aspects-of-deep-learning) — configuring your ML pipeline, regularization techniques, and initialization strategies
- [Advanced optimization methods](guides/02-tuning-and-optimization.md#optimization-algorithms) — momentum, RMSProp, Adam, and learning rate schedules
- [Hyperparameter tuning & batch normalization](guides/02-tuning-and-optimization.md#hyperparameter-tuning-batch-normalization-and-programming-frameworks) — systematic search, normalizing activations, and choosing frameworks

</details>

📓 **Rendered notebook:** [`optimization-techniques.ipynb`](rendered-notes/optimization-techniques.ipynb)

---

### 🎯 Module 3 — Strategically Structuring ML Initiatives

> *Learn the decision-making framework that separates successful ML projects from those that stall.*

![](https://i.pinimg.com/originals/9b/fa/97/9bfa978a4cf40fe2cdf8c710deb9b6f9.png)

<details>
<summary>🔍 What's covered in this module?</summary>

- [Setting targets and measuring progress](guides/03-project-strategy-playbook.md#ml-strategy-1) — orthogonalization, satisficing vs. optimizing metrics, and human-level baselines
- [Systematic error diagnosis](guides/03-project-strategy-playbook.md#ml-strategy-2) — ceiling analysis, handling mismatched distributions, and multi-task / transfer learning tradeoffs

</details>

🧪 **Practice quizzes:**
- [Quiz 1 — Avian Classification Scenario](labs/ml-strategy-quizzes/quiz-01-avian-classification-scenario.md)
- [Quiz 2 — Self-Driving Car Scenario](labs/ml-strategy-quizzes/quiz-02-self-driving-car-scenario.md)

📓 **Rendered notebook:** [`ml-project-structure.ipynb`](rendered-notes/ml-project-structure.ipynb)

---

### 🔬 Module 4 — Convolutional Networks for Visual Understanding

> *See how machines see — from pixel-level filters to state-of-the-art image recognition and beyond.*

| Convolution in Action | Feature Map Extraction | Pooling & Architecture |
|:---------------------:|:---------------------:|:---------------------:|
| ![](https://i.stack.imgur.com/9OZKF.gif) | ![](https://cdn-images-1.medium.com/max/600/1*GdxHFaUDbvTXJreKg3S8SQ.gif) | ![](https://www.guru99.com/images/tensorflow/082918_1325_ConvNetConv9.gif) |

<details>
<summary>🔍 What's covered in this module?</summary>

- [CNN building blocks](guides/04-visual-recognition-networks.md#foundations-of-cnns) — convolutions, padding, stride, and pooling
- [Landmark architectures](guides/04-visual-recognition-networks.md#deep-convolutional-models-case-studies) — LeNet, AlexNet, VGG, ResNet, Inception
- [Detecting objects in images](guides/04-visual-recognition-networks.md#object-detection) — YOLO, bounding boxes, non-max suppression
- [Face verification & artistic style transfer](guides/04-visual-recognition-networks.md#special-applications-face-recognition--neural-style-transfer) — Siamese networks, FaceNet, neural style transfer

</details>

<details>
<summary>📄 Recommended research papers</summary>

- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
- [YOLO9000](https://arxiv.org/pdf/1612.08242.pdf)
- [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
- [FaceNet: A Unified Embedding for Face Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

</details>

📓 **Rendered notebook:** [`convnets-explained.ipynb`](rendered-notes/convnets-explained.ipynb)

---

### 🔮 Module 5 — Processing Sequential & Temporal Data

> *From language translation to speech recognition — master the architectures that handle ordered data.*

![](https://3.bp.blogspot.com/-3Pbj_dvt0Vo/V-qe-Nl6P5I/AAAAAAAABQc/z0_6WtVWtvARtMk0i9_AtLeyyGyV6AI4wCLcB/s1600/nmt-model-fast.gif)

---

| Architecture | Visualization |
|:------------:|:-------------:|
| 🔁 **Vanilla RNN** | ![](https://cdn-images-1.medium.com/max/880/1*xn5kA92_J5KLaKcP7BMRLA.gif) |
| 🧱 **LSTM** | ![](https://cdn-images-1.medium.com/max/880/1*goJVQs-p9kgLODFNyhl9zA.gif) |
| 🚪 **GRU** | ![](https://cdn-images-1.medium.com/max/880/1*FpRS0C3EHQnELVaWRvb8bg.gif) |

<details>
<summary>🔍 What's covered in this module?</summary>

- [Recurrent neural network architectures](guides/05-temporal-sequence-learning.md#recurrent-neural-networks) — vanishing gradients, bidirectional RNNs, deep RNNs
- [NLP and distributed word representations](guides/05-temporal-sequence-learning.md#natural-language-processing--word-embeddings) — Word2Vec, GloVe, sentiment analysis
- [Attention and sequence-to-sequence models](guides/05-temporal-sequence-learning.md#sequence-models--attention-mechanism) — encoder-decoder, beam search, Bleu score

</details>

📓 **Rendered notebook:** [`sequence-modeling.ipynb`](rendered-notes/sequence-modeling.ipynb)

---

## 📺 Video Lecture Playlists

<details>
<summary><strong>🎬 Expand to see all YouTube playlists</strong></summary>

| 🏷️ Series | 🔗 Playlist |
|:----------|:-----------|
| Machine Learning (Stanford) | [Andrew Ng — Full ML Course](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) |
| DL Course 1 | [Neural Networks & Deep Learning](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0) |
| DL Course 2 | [Hyperparameter Tuning, Regularization & Optimization](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc) |
| DL Course 3 | [Structuring Machine Learning Projects](https://www.youtube.com/playlist?list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b) |
| DL Course 4 | [Convolutional Neural Networks](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) |
| DL Course 5 | [Sequence Models](https://www.youtube.com/playlist?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6) |
| CS230 | [Deep Learning — Stanford Autumn 2018](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb) |

</details>

---

## 📚 Supplementary Materials

| Resource | Description |
|:---------|:-----------|
| 📄 [Comprehensive DL Reference (PDF)](resources/comprehensive-dl-reference.pdf) | All five courses condensed into a single reference document |
| 📓 [Andrew Ng's ML Notebooks](https://github.com/ishandutta2007/Andrew-NG-Coursera-Deeplearning/tree/master/Machine%20Learning%20notebooks%20By%20Andrew%20NG) | Original machine learning notebook collection |

---

## ✅ Study Checklist

Track your progress through the specialization:

- [ ] 📐 Complete Module 1 — Neural Network Foundations
- [ ] ⚡ Complete Module 2 — Tuning & Optimization
- [ ] 🎯 Complete Module 3 — ML Strategy
- [ ] 🔬 Complete Module 4 — Convolutional Networks
- [ ] 🔮 Complete Module 5 — Sequence Models
- [ ] 📝 Finish all quizzes in `labs/`
- [ ] 💻 Submit all assignments in `assignments/`
- [ ] 📖 Read recommended research papers

---

<p align="center">
  <strong>🧠 Keep exploring, keep learning — every gradient step counts! 🚀</strong>
</p>
