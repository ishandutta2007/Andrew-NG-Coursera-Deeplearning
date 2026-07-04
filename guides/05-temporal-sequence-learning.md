
# 🔄 Temporal Sequence Learning Models

This guide covers the fifth and concluding course in the deep learning specialization offered on [Coursera](https://www.coursera.org/specializations/deep-learning), developed by [deeplearning.ai](http://deeplearning.ai/) and instructed by Andrew Ng.

# 📚 Guide Collection

* [**Foundations of Neural Nets**](guides/01-foundations-of-neural-nets.md)
* [**Tuning & Optimization**](guides/02-tuning-and-optimization.md)
* [**Project Strategy Playbook**](guides/03-project-strategy-playbook.md)
* [**Visual Recognition Networks**](guides/04-visual-recognition-networks.md)
* [**Temporal Sequence Learning**](guides/05-temporal-sequence-learning.md)

## 📑 Table of Contents

* [Temporal Sequence Learning Models](#-temporal-sequence-learning-models)
   * [Table of Contents](#-table-of-contents)
   * [Overview of the Course](#-overview-of-the-course)
   * [Recurrent Neural Networks (RNNs)](#-recurrent-neural-networks-rnns)
      * [The Case for Sequential Models](#-the-case-for-sequential-models)
      * [Symbols and Conventions](#-symbols-and-conventions)
      * [Architecture of Recurrent Networks](#-architecture-of-recurrent-networks)
      * [Temporal Backpropagation](#-temporal-backpropagation)
      * [RNN Architectural Variants](#-rnn-architectural-variants)
      * [Language Modeling and Sequence Generation](#-language-modeling-and-sequence-generation)
      * [Generating New Sequences via Sampling](#-generating-new-sequences-via-sampling)
      * [The Vanishing Gradient Challenge in RNNs](#-the-vanishing-gradient-challenge-in-rnns)
      * [Gated Recurrent Units (GRU)](#-gated-recurrent-units-gru)
      * [Long Short-Term Memory Networks (LSTM)](#-long-short-term-memory-networks-lstm)
      * [Bidirectional Recurrent Networks](#-bidirectional-recurrent-networks)
      * [Stacked (Deep) RNNs](#-stacked-deep-rnns)
      * [Backward Pass in Recurrent Networks](#-backward-pass-in-recurrent-networks)
   * [NLP and Word Embeddings](#-nlp-and-word-embeddings)
      * [Getting Started with Word Embeddings](#-getting-started-with-word-embeddings)
         * [How Words Are Represented](#-how-words-are-represented)
         * [Leveraging Word Embeddings](#-leveraging-word-embeddings)
         * [Characteristics of Word Embeddings](#-characteristics-of-word-embeddings)
         * [The Embedding Matrix](#-the-embedding-matrix)
      * [Training Word Embeddings: Word2Vec and GloVe](#-training-word-embeddings-word2vec-and-glove)
         * [Approaches to Learning Embeddings](#-approaches-to-learning-embeddings)
         * [The Word2Vec Algorithm](#-the-word2vec-algorithm)
         * [Negative Sampling Technique](#-negative-sampling-technique)
         * [GloVe Vectors](#-glove-vectors)
      * [Practical Uses of Word Embeddings](#-practical-uses-of-word-embeddings)
         * [Classifying Sentiment](#-classifying-sentiment)
         * [Removing Bias from Word Embeddings](#-removing-bias-from-word-embeddings)
   * [Sequence-to-Sequence Models and Attention](#-sequence-to-sequence-models-and-attention)
      * [Encoder-Decoder Architectures](#-encoder-decoder-architectures)
         * [Foundational Models](#-foundational-models)
         * [Selecting the Optimal Output Sentence](#-selecting-the-optimal-output-sentence)
         * [Beam Search Algorithm](#-beam-search-algorithm)
         * [Improving Beam Search](#-improving-beam-search)
         * [Diagnosing Errors in Beam Search](#-diagnosing-errors-in-beam-search)
         * [BLEU Score Evaluation](#-bleu-score-evaluation)
         * [Intuition Behind Attention](#-intuition-behind-attention)
         * [The Attention Mechanism in Detail](#-the-attention-mechanism-in-detail)
      * [Audio and Speech Processing](#-audio-and-speech-processing)
         * [Automatic Speech Recognition](#-automatic-speech-recognition)
         * [Wake Word / Trigger Word Detection](#-wake-word--trigger-word-detection)
   * [Supplementary Material](#-supplementary-material)
      * [Machine Translation with Attention (Notebook Walkthrough)](#-machine-translation-with-attention-notebook-walkthrough)


## 🎯 Overview of the Course

Here is the official course description from the [course page](https://www.coursera.org/learn/nlp-sequence-models):

> This course will teach you how to build models for natural language, audio, and other sequence data. Thanks to deep learning, sequence algorithms are working far better than just two years ago, and this is enabling numerous exciting applications in speech recognition, music synthesis, chatbots, machine translation, natural language understanding, and many others. 
>
> You will:
> - Understand how to build and train Recurrent Neural Networks (RNNs), and commonly-used variants such as GRUs and LSTMs.
> - Be able to apply sequence models to natural language problems, including text synthesis. 
> - Be able to apply sequence models to audio applications, including speech recognition and music synthesis.
>
> This is the fifth and final course of the Deep Learning Specialization.



## 🔁 Recurrent Neural Networks (RNNs)

> Dive into recurrent neural networks — a family of architectures proven to excel at processing temporal and sequential data. This section walks through several important variants, including LSTMs, GRUs, and Bidirectional RNNs.

### 🌊 The Case for Sequential Models

- Architectures like RNNs and LSTMs have revolutionized how we handle sequential information in recent years.
- Here are some real-world applications involving sequential data:
  - Speech recognition (**sequence to sequence**):
    - X: audio waveform
    - Y: transcribed text
  - Music generation (**one to sequence**):
    - X: empty input or a single integer
    - Y: audio waveform
  - Sentiment classification (**sequence to one**):
    - X: text input
    - Y: rating value from one to five
  - DNA sequence analysis (**sequence to sequence**):
    - X: DNA base sequence
    - Y: annotation labels
  - Machine translation (**sequence to sequence**):
    - X: sentence in the source language
    - Y: sentence in the target language
  - Video activity recognition (**sequence to one**):
    - X: sequence of video frames
    - Y: activity label
  - Name entity recognition (**sequence to sequence**):
    - X: text input
    - Y: entity labels for each token
    - Search engines leverage this to index various types of words in a document.
- Every one of these problems — whether the input/output is a sequence or not — can be framed as a supervised learning task with labeled training pairs (X, Y).

### 🏷️ Symbols and Conventions

- This section lays out the notation used throughout the course.
- **Illustrative example**:
  - Named entity recognition scenario:
    - X: "Harry Potter and Hermoine Granger invented a new spell."
    - Y:   1   1   0   1   1   0   0   0   0
    - Both sequences have length 9. A value of 1 indicates a name; 0 means it is not a name.
- We index the first element of x as x<sup><1></sup>, the second as x<sup><2></sup>, and so on.
  - x<sup><1></sup> = Harry
  - x<sup><2></sup> = Potter
- Likewise, y is indexed as y<sup><1></sup>, y<sup><2></sup>, etc.
  - y<sup><1></sup> = 1
  - y<sup><2></sup> = 1

- T<sub>x</sub> denotes the length of the input sequence, and T<sub>y</sub> denotes the length of the output sequence.
  
  - In the example above, T<sub>x</sub> = T<sub>y</sub> = 9, though they can differ in other tasks.
- x<sup>(i)\<t></sup> refers to the t-th element of the i-th input example. Similarly, y<sup>(i)\<t></sup> is the t-th element of the output for the i-th training example.
- T<sub>x</sub><sup>(i)</sup> is the input sequence length for training example i (which may vary between examples). The same goes for T<sub>y</sub><sup>(i)</sup>.

- **Encoding words**:
    - Throughout this course we work with **NLP** (natural language processing). A central challenge in NLP is: how do we represent a word numerically?

    1. First, construct a **vocabulary** — a list of all words in your corpus.
        - Example:
            - [a ... And   ... Harry ... Potter ... Zulu]
            - Every word gets a unique index.
            - The list is sorted alphabetically here.
        - Modern vocabularies typically range from 30,000 to 50,000 entries. 100,000 is common, and some large organizations use vocabularies exceeding a million.
        - You can build a vocabulary by scanning your text data and keeping the m most frequently occurring words, or by looking up the most common m words online.
    2. Then create a **one-hot encoding** for every word in your dataset based on this vocabulary.
        - What happens if you encounter a word not in your dictionary?
        - Add a special `<UNK>` (unknown) token to the vocabulary and use its index for the one-hot vector.
    - Complete illustration:   
        ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//01.png)

- The objective is to learn a mapping from this representation of x to the target output y via supervised learning with a sequence model.

### 🧱 Architecture of Recurrent Networks

- Why can't a standard neural network handle sequence tasks well? Two core issues:
  - Input and output lengths can vary across different examples.
    - Padding to the maximum length is a workaround for regular NNs, but it's an inelegant solution.
  - It doesn't share learned features across different positions in the text/sequence.
    - Feature sharing (similar to what CNNs do) can dramatically reduce the parameter count. That's exactly what RNNs achieve.
- Recurrent neural networks sidestep both of these problems.
- Let's construct an RNN to tackle the **name entity recognition** task:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//02.png)
  - In this particular problem T<sub>x</sub> = T<sub>y</sub>. For cases where they differ, the RNN architecture is adjusted accordingly.
  - a<sup><0></sup> is typically initialized to zeros, though random initialization is occasionally used.
  - Three weight matrices are involved: W<sub>ax</sub>, W<sub>aa</sub>, and W<sub>ya</sub> with dimensions:
    - W<sub>ax</sub>: (NoOfHiddenNeurons, n<sub>x</sub>)
    - W<sub>aa</sub>: (NoOfHiddenNeurons, NoOfHiddenNeurons)
    - W<sub>ya</sub>: (n<sub>y</sub>, NoOfHiddenNeurons)
- The weight matrix W<sub>aa</sub> acts as the memory mechanism through which the RNN preserves information from earlier layers.
- Many papers and textbooks depict the same architecture in this compressed form:  
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//03.png)
  - This compact form can be harder to parse. It's generally more intuitive to unroll the diagram.
- In the architecture described, the current prediction y&#770;<sup>\<t></sup> depends on all preceding inputs and activations.
- Consider the sentence 'He Said, "Teddy Roosevelt was a great president"'. Here, "Teddy" is a person's name, but we deduce that from the word **president** that appears after it, not from **He** and **said** which precede it.
- A key limitation of this architecture is that it cannot leverage information from later positions in the sequence. We'll address this with **Bidirectional RNNs** (BRNN) later.
- Forward propagation equations for this architecture:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//04.png)
  - The activation function for `a` is typically tanh or ReLU; for `y` it depends on the task — sigmoid and softmax are common choices. For name entity recognition, sigmoid is used since there are only two classes.
- To facilitate more complex RNN designs, these equations can be expressed more compactly.
- **Condensed RNN notation**:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//05.png)
  - w<sub>a</sub> is the horizontal concatenation of w<sub>aa</sub> and w<sub>ax</sub>.
  - [a<sup>\<t-1></sup>, x<sup>\<t></sup>] is the vertical stack of a<sup>\<t-1></sup> and x<sup>\<t></sup>.
  - w<sub>a</sub> shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n<sub>x</sub>)
  - [a<sup>\<t-1></sup>, x<sup>\<t></sup>] shape: (NoOfHiddenNeurons + n<sub>x</sub>, 1)

### ⏪ Temporal Backpropagation

- Let's examine how backpropagation operates within an RNN.
- Most deep learning frameworks handle backpropagation automatically, but understanding the mechanics in RNNs is valuable.
- Here is the computation graph:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//06.png)
  - Note that w<sub>a</sub>, b<sub>a</sub>, w<sub>y</sub>, and b<sub>y</sub> are shared across all time steps in the sequence.
- The cross-entropy loss function is used:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//07.png)
  - The first equation gives the loss for a single example; the total sequence loss is the sum of all individual losses.
- Computation graph with losses visualized:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//08.png)
- This process is termed **backpropagation through time** because the activation `a` is propagated from one time step to the previous one — effectively moving backward through time.

### 🔀 RNN Architectural Variants

- So far we have only looked at one RNN configuration where T<sub>x</sub> equals T<sub>Y</sub>. Since this isn't always the case, alternative architectures are needed.
- The taxonomy in this section draws inspiration from Andrej Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). This image summarizes all the types:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//09.jpg)
- The architecture we covered earlier is classified as **Many to Many**.
- For sentiment analysis — where X is text and Y is a rating from 1 to 5 — the appropriate architecture is **Many to One**, as illustrated by Karpathy.   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//10.png)
- A **One to Many** design is suitable for tasks like music generation.  
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//11.png)
  - Note that from the second time step onward, the generated output is fed back as input to the network.
- There's another important variant within **Many To Many**. In tasks like machine translation, input and output sequences usually differ in length. An encoder-decoder _Many To Many_ architecture addresses this:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//12.png)
  - This architecture splits into an encoder and a decoder. The encoder compresses the input into a single representation, which the decoder then transforms into the output. Each part has its own set of weights.
- Summary of all RNN configurations:   
   ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//12_different_types_of_rnn.jpg)
- There is also the **attention** architecture, which we'll explore in a later chapter.

### 📖 Language Modeling and Sequence Generation

- RNNs are exceptionally effective for language modeling tasks. Here we'll build one from scratch.
- **What exactly is a language model?**
  - Imagine a speech recognition system hears a sentence that could be interpreted two ways:
    - The apple and **pair** salad
    - The apple and **pear** salad
  - Since **pair** and **pear** are phonetically identical, how does the system choose?
  - A language model assigns probabilities to both sentences, and the system picks the one with the higher likelihood.
- A language model's job is to compute the probability of any given word sequence.
- **Constructing a language model with RNNs:**
  - Start by obtaining a **training set**: a large body of text in the target language.
  - Tokenize this corpus by defining a vocabulary and one-hot encoding each word.
  - Include an end-of-sentence token `<EOS>` in the vocabulary and append it to each sentence. Use `<UNK>` for out-of-vocabulary words.
- Take the sentence "Cats average 15 hours of sleep a day. `<EOS>`"
  - During training, the process looks like this:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//13.png)
  - The loss is computed using cross-entropy:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//14.png)
    - `i` iterates over all items in the corpus; `t` iterates over all time steps.
- Using the trained model:
  1.  To predict the **next word**, feed the sentence through the RNN and examine the final y<sup>^\<t></sup> output vector — the highest-probability word wins.
  2.  To compute the **probability of an entire sentence**, apply the chain rule:
      - p(y<sup><1></sup>, y<sup><2></sup>, y<sup><3></sup>) = p(y<sup><1></sup>) * p(y<sup><2></sup> | y<sup><1></sup>) * p(y<sup><3></sup> | y<sup><1></sup>, y<sup><2></sup>)
      - This is just feeding the sentence through the RNN and multiplying the output probabilities.

### 🎲 Generating New Sequences via Sampling

- Once a sequence model has been trained as a language model, you can verify what it's learned by sampling novel sequences from it.
- Here's how to sample a new sequence from a trained model:
  1. Start with this model:   
     ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//15.png)
  2. Initialize with a<sup><0></sup> = zero vector, and x<sup><1></sup> = zero vector.
  3. Randomly select a word from the probability distribution given by y&#770;<sup><1></sup>. It might be "The", for instance.
     - In numpy: `numpy.random.choice(...)`
     - This step introduces randomness — each run produces a different starting word.
  4. Feed the chosen word along with the computed a<sup><1></sup> into the next time step.
  5. Continue steps 3 & 4 either for a predetermined number of steps or until the `<EOS>` token appears.
  6. You may choose to filter out any `<UNK>` tokens from the generated output.
- Everything described above builds a word-level language model. You can also create a **character-level** variant.
- In a character-level language model, the vocabulary consists of `[a-zA-Z0-9]`, punctuation marks, special characters, and optionally the <EOS> token.
- Trade-offs of character-level versus word-level models:
  - ✅ Advantages:
    1. No `<UNK>` tokens — the model can produce any word.
  - ❌ Disadvantages:
    1. Sequences become significantly longer.
    2. Character-level models are weaker at capturing long-range dependencies.
    3. They are also more computationally demanding and harder to train.
- Andrew's observation: word-level models remain the mainstream choice, but as hardware speeds up, character-level models are gaining traction in specific domains — especially where unknown or specialized vocabulary is common.

### 📉 The Vanishing Gradient Challenge in RNNs

- A major hurdle for basic RNNs is the **vanishing gradient** problem.

- An RNN processing 10,000 time steps effectively has 10,000 layers, making optimization extremely difficult.

- Consider this language modeling example with two sentences the model must learn:

  - "The **cat**, which already ate ..., **was** full"
  - "The **cats**, which already ate ..., **were** full"
  - The dots represent many intervening words.

- The model needs to learn that "was" pairs with "cat" and "were" pairs with "cats". Basic RNNs struggle to capture such long-range dependencies.

- As we discussed with deep neural networks, very deep architectures suffer from vanishing gradients. The same applies to RNNs with long sequences.   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//16.png)   
  - To compute the output for "was", gradients must flow all the way back through the network. Repeatedly multiplying small fractions causes gradients to vanish; multiplying large numbers causes them to explode.
  - Consequently, some weights receive inadequate updates.

- Practically, this means the network has difficulty remembering "cat" by the time it reaches "was". It can't reliably determine the correct verb form (was/were) based on the distant subject (cat/cats).

- The bottom line: basic RNNs are poor at maintaining **long-term dependencies**.

- > In theory, RNNs are absolutely capable of handling such "long-term dependencies." A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don't seem to be able to learn them. http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- The _vanishing gradient_ issue generally poses a bigger challenge than _exploding gradients_ for RNNs. Solutions are covered in the following sections.

- Exploding gradients are easy to detect — your weights turn into `NaN`. A common fix is **gradient clipping**: if the gradient magnitude exceeds a threshold, rescale the gradient vector. Gradients are capped at a specified maximum value.

  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//26.png)

- **Additional notes**:
  - Remedies for exploding gradients:
    - Truncated backpropagation.
      - Skip updating some of the earlier weights.
      - Not an ideal approach since some weights go un-updated.
    - Gradient clipping.
  - Remedies for vanishing gradients:
    - Careful weight initialization.
      - Such as He initialization.
    - Echo state networks.
    - Use LSTM/GRU architectures.
      - By far the most popular solution.
      - Discussed next.

### 🚪 Gated Recurrent Units (GRU)

- The GRU is an RNN variant designed to mitigate the vanishing gradient problem and retain long-range information.

- A basic RNN cell can be visualized as:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//17.png)

- We'll use a similar diagrammatic style for the GRU.

- Each GRU layer introduces a new variable `C` — the memory cell. It determines whether to remember or forget information.

- In GRUs, C<sup>\<t></sup> = a<sup>\<t></sup>

- GRU equations:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//18.png)
  - The update gate outputs a value between 0 and 1.
    - For intuition, think of the gate as being nearly 0 or 1 most of the time.
  - The memory cell is updated based on the gate value and the previous cell state.

- Applying this to the cat/was example:

  - Sentence: "The **cat**, which already ate ........................, **was** full"

  - Imagine U as a binary flag indicating whether a singular noun should be stored in memory.

  - Tracking C and U across the sentence:

    - | Word    | Update gate(U)             | Cell memory (C) |
      | ------- | -------------------------- | --------------- |
      | The     | 0                          | val             |
      | cat     | 1                          | new_val         |
      | which   | 0                          | new_val         |
      | already | 0                          | new_val         |
      | ...     | 0                          | new_val         |
      | was     | 1 (I don't need it anymore)| newer_val       |
      | full    | ..                         | ..              |
- GRU schematic:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//19.png)
  
  - Visual diagrams in the style of http://colah.github.io/posts/2015-08-Understanding-LSTMs/ are widely used for understanding GRUs and LSTMs, though Andrew Ng prefers focusing on the equations.
- Since the update gate U is typically a very small value (e.g., 0.00001), GRUs effectively avoid the vanishing gradient problem.
  
  - In the equation, this results in C<sup>\<t></sup> = C<sup>\<t-1></sup> in many cases.
- Tensor shapes:
  - a<sup>\<t></sup> shape is (NoOfHiddenNeurons, 1)
  - c<sup>\<t></sup> is the same as a<sup>\<t></sup>
  - c<sup>~\<t></sup> is the same as a<sup>\<t></sup>
  - u<sup>\<t></sup> is also the same dimensions of a<sup>\<t></sup>
- All multiplications in the equations are element-wise.
- What we've described above is the simplified GRU. The full version adds a **relevance gate** used when computing the candidate C. This gate controls how much C<sup>\<t-1></sup> influences C<sup>\<t></sup>.
  - Full GRU equations:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//20.png)
  - Tensor shapes remain the same.
- Why these particular architectures? Researchers have tested countless variations over the years, all aimed at solving the vanishing gradient problem. The full GRU has emerged as one of the most reliable and broadly applicable RNN designs. You're free to experiment, but GRUs and LSTMs are the established standards.

### 🧠 Long Short-Term Memory Networks (LSTM)

- The LSTM is another RNN architecture capable of capturing long-term dependencies. It's more powerful and general-purpose than the GRU.
- In LSTMs, C<sup>\<t></sup> != a<sup>\<t></sup>
- LSTM unit equations:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//21.png)
- Comparing GRU and LSTM gates: GRUs have an update gate `U`, a relevance gate `r`, and a candidate cell variable C<sup>\~\<t></sup>. LSTMs have an update gate `U` (also called the input gate I), a forget gate `F`, an output gate `O`, and a candidate cell variable C<sup>\~\<t></sup>.
- Schematic (inspired by http://colah.github.io/posts/2015-08-Understanding-LSTMs/):    
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//22.png)
- Notable LSTM variants include:
  - LSTM with **peephole connections**.
    - The standard LSTM modified so that C<sup>\<t-1></sup> feeds into every gate.
- There is no universally superior choice between LSTMs and their variants. GRUs have the advantage of simplicity, enabling larger networks, while LSTMs offer greater expressiveness and generality.

### ↔️ Bidirectional Recurrent Networks

- Several techniques exist for building more capable sequence models. Two key ones are bidirectional RNNs and deep (stacked) RNNs.
- Recall the name entity recognition example:  
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//23.png)
- The name **Teddy** can't be inferred from **He** and **said** alone — the clue comes from the later word **bears**.
- Bidirectional RNNs (BiRNNs) solve this problem.
- BiRNN architecture:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//24.png)
- Important: the BiRNN forms an **acyclic graph**.
- One direction of forward propagation runs left-to-right; the other runs right-to-left. The network learns from both directions simultaneously.
- Predictions use y&#770;<sup>\<t></sup>, which combines activations arriving from both the left and right passes.
- The individual cells can be basic RNN units, LSTMs, or GRUs.
- For most NLP and text-processing applications, a BiRNN with LSTM cells is the standard choice.
- The main drawback of BiRNNs is that the complete input sequence must be available before processing can begin. For real-time applications like live speech recognition, you'd have to wait for the speaker to finish before generating predictions.

### 📚 Stacked (Deep) RNNs

- A single-layer RNN is often sufficient. However, for certain complex tasks, stacking multiple RNN layers into a deeper architecture is beneficial.
- A three-layer deep RNN looks like this:  
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//25.png)
- Whereas feed-forward networks can have 100 or 200+ layers, even 3 stacked RNN layers is considered deep and computationally expensive.
- Sometimes, additional feed-forward layers are appended after the recurrent cells.


### 🔙 Backward Pass in Recurrent Networks

- > In modern deep learning frameworks, you only have to implement the forward pass, and the framework takes care of the backward pass, so most deep learning engineers do not need to bother with the details of the backward pass. If however you are an expert in calculus and want to see the details of backprop in RNNs, you can work through this optional portion of the notebook.

- The above quote comes from this [notebook](https://www.coursera.org/learn/nlp-sequence-models/notebook/X20PE/building-a-recurrent-neural-network-step-by-step). Refer to it for a detailed programming-oriented walkthrough of RNN backpropagation.

## 💬 NLP and Word Embeddings

> Deep learning applied to natural language processing is a transformative combination. By using word vector representations and embedding layers, you can train recurrent networks that achieve outstanding results across many industries. Common applications include sentiment analysis, named entity recognition, and machine translation.

### 🌐 Getting Started with Word Embeddings

#### 🔤 How Words Are Represented

- Deep learning, particularly RNNs and their deeper variants, has fundamentally transformed NLP.
- Word embeddings provide a way to represent words numerically, enabling algorithms to automatically discover analogies (e.g., "king" is to "queen" as "man" is to "woman").
- Previously, we represented words using a vocabulary and one-hot vectors.
  - Visual example:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//27.png)
  - We use the notation **O** <sub>idx</sub> for a one-hot encoded word as shown above.
  - A fundamental weakness of one-hot encoding is that it treats every word as an isolated entity, preventing the algorithm from generalizing across semantically related words.
    - Example: "I want a glass of **orange** ______" — a model should predict **juice**.
    - A similar input: "I want a glass of **apple** ______" — the model may struggle to predict **juice** if it wasn't explicitly trained on this phrase. The two contexts aren't connected even though orange and apple are semantically similar.
  - The inner product of any two one-hot vectors is zero, and all pairwise distances are identical.
- Wouldn't it be preferable to learn a feature-rich representation for words like man, woman, king, queen, apple, and orange?   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//28.png)
  - Each word is described by, say, 300 floating-point features.
  - Each word's column becomes a 300-dimensional vector serving as its representation.
  - We use the notation **e**<sub>5391</sub> to refer to the feature vector for the word **man**.
  - Revisiting our examples:
    -  "I want a glass of **orange** ______" 
    -  I want a glass of **apple** ______
  - Orange and apple now share many similar features, making it far easier for the algorithm to transfer knowledge between them.
  - This representation is called **Word embeddings**.
- To visualize word embeddings, we apply the t-SNE algorithm to project the high-dimensional features down to 2D:    
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//29.png)
  - Semantically related words cluster together in the visualization.
- The term **word embeddings** comes from the idea of embedding each word as a unique vector within an n-dimensional space.

#### 🔗 Leveraging Word Embeddings

- Let's see how the learned feature representations can be applied to the Named Entity Recognition task.
- Consider this example:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//30.png)
- **Sally Johnson** is identified as a person's name.
- Having trained on this sentence, the model should be able to recognize that "**Robert Lin** is an *apple* farmer" also contains a person's name, since apple and orange have similar embeddings.
- Even for the sentence "**Mahmoud Badry** is a *durian* cultivator", the model should correctly identify the name — even if the word *durian* never appeared in the training data. This demonstrates the power of word embeddings.
- Word embedding algorithms can process billions of words from unlabeled text (e.g., 100 billion words) to learn these representations.
- Transfer learning workflow for word embeddings:
  1. Learn word embeddings from a massive text corpus (1–100 billion words).
     - Alternatively, download pre-trained embeddings.
  2. Transfer these embeddings to your new task, which has a much smaller training set (e.g., ~100k words).
  3. Optionally, continue fine-tuning the embeddings with your new data.
     - This is worthwhile only if your task-specific dataset is sufficiently large.
- Word embeddings deliver the greatest benefit when your target task has a relatively small labeled dataset.
- Another advantage: embeddings dramatically reduce input dimensionality!
  - Compare a 10,000-dimensional one-hot vector versus a compact 300-dimensional embedding.
- There's a fascinating parallel between word embeddings and face recognition:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//31.png)
  - In face recognition, each face is encoded as a vector, and similarity is measured between vectors.
  - The terms **encoding** and **embeddings** are used interchangeably in this context.
- In word embedding tasks, we learn a fixed representation for every word in our vocabulary (unlike image encoding, where each new image must be mapped to an n-dimensional vector). The learning algorithms are explored in subsequent sections.

#### 🧲 Characteristics of Word Embeddings

- One of the most remarkable properties of word embeddings is their ability to support analogy reasoning. While analogy tasks aren't the most practical NLP application, they illustrate the power of embeddings.
- Analogy example:
  - Given this embedding table:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//32.png)
  - Can we solve this analogy?
    - Man ==> Woman
    - King ==> ??
  - Subtracting e<sub>Man</sub> from e<sub>Woman</sub> yields the vector `[-2  0  0  0]`
  - Likewise, e<sub>King</sub> - e<sub>Queen</sub> = `[-2  0  0  0]`
  - The difference in both cases encodes gender.   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//33.png)
    - This vector captures the gender dimension.
    - Note: this is a 2D t-SNE projection of a 4D vector — it's for visualization only. Don't use t-SNE to identify analogies.
  - Reformulating the problem:
    - e<sub>Man</sub> - e<sub>Woman</sub> ≈ e<sub>King</sub> - e<sub>??</sub>
  - Expressed mathematically:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//34.png)
  - The best match turns out to be e<sub>Queen</sub> — the vector that most closely satisfies the relation.
- Cosine similarity — the standard similarity metric:
  - Formula:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//35.png)
    - $$\text{CosineSimilarity(u, v)} = \frac {u . v} {||u||_2 ||v||_2} = cos(\theta)$$
    - The numerator is the dot product of `u` and `v`. It grows larger when the vectors are more aligned.
- Euclidean distance can also serve as a similarity function (though it measures dissimilarity, so negate it).
- This equation can be applied to solve the analogy problem where `u` = e<sub>w</sub> and `v` = e<sub>king</sub> - e<sub>man</sub> + e<sub>woman</sub>

#### 🗂️ The Embedding Matrix

- When you train a word embedding algorithm, the result is an **<u>embedding matrix</u>**.
- Example walkthrough:
  - Suppose our vocabulary contains 10,000 words (plus the <UNK> token).
  - The algorithm produces a matrix `E` with shape (300, 10000) — assuming 300 features per word.   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//36.png)
  - If O<sub>6257</sub> is the one-hot encoding for **orange** with shape (10000, 1), then   
    _np.dot(`E`,O<sub>6257</sub>) = e<sub>6257</sub>_ with shape (300, 1).
  - Generally, _np.dot(`E`, O<sub>j</sub>) = e<sub>j</sub>_
- In practice, `E` is initialized randomly and its entries are learned during training.
- Using matrix multiplication to extract embeddings is inefficient. Instead, column slicing is used. Keras provides a dedicated embedding layer that performs this lookup directly without multiplication.

### 🎓 Training Word Embeddings: Word2Vec and GloVe

#### 📘 Approaches to Learning Embeddings

- Let's survey algorithms that learn word embeddings.
- Early approaches were complex, but they progressively simplified over time.
- We'll begin with the more involved methods to build intuition.
- **<u>Neural language model</u>**:
  - Starting example:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//37.png)
  - The goal is to build a model that predicts the next word.
  - The neural network used for this:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//38.png)
    - We obtain e<sub>j</sub> via `np.dot(`E`,o<sub>j</sub>)`
    - The hidden layer has parameters `W1` and `b1`; the softmax layer has `W2` and `b2`.
    - Input dimension is (300*6, 1) when using a context window of 6 preceding words.
    - The optimization targets are the `E` matrix and all layer parameters. The objective is to maximize the likelihood of predicting the next word given the preceding context.
  - This model, introduced in 2003, produces quite good word embeddings.
- In the above example, we used a 6-word lookback window. Other context strategies exist when learning embeddings:
  - Consider the sentence: "I want a glass of orange **juice** to go along with my cereal"
  - To predict **juice**, possible **context** choices include:
    1. Previous 4 words.
       - Using a window of the last 4 words (4 is a hyperparameter): "<u>a glass of orange</u>", then predict the next word.
    2. 4 words to the left and 4 to the right.
       - "<u>a glass of orange</u>" and "<u>to go along with</u>"
    3. Just the immediately preceding word.
       - "<u>orange</u>"
    4. A single nearby word.
       - "<u>glass</u>" is near juice.
       - This is the foundation of the **skip-gram** model.
       - It's surprisingly simple and remarkably effective.
       - Covered in the next section.
- Key insight from researchers: for building a _language model_ specifically, using the previous few words as context is natural. But if your primary aim is to learn _word embeddings_, any of the other context strategies work well and produce meaningful embeddings.
- In summary: posing language modeling as a supervised learning problem — predicting target words from context — enables the learning of high-quality word embeddings.

#### ⚡ The Word2Vec Algorithm

- Before diving into Word2Vec, let's understand **skip-grams**:
  - Example sentence: "I want a glass of orange juice to go along with my cereal"
  - We select a **context** word and a **target** word.
  - The target is randomly chosen within a window around the context word.
  
    | Context | Target | How far |
    | ------- | ------ | ------- |
    | orange  | juice  | +1      |
    | orange  | glass  | -2      |
    | orange  | my     | +6      |

    This converts the task into a supervised learning problem.
  - Predicting targets within a ±10-word window (10 is just an example) is challenging.
  - The purpose is to learn the word embedding model through this process.
- The Word2Vec model:
  - Vocabulary size = 10,000 words
  - Let `c` denote the context word and `t` the target word.
  - The goal is to learn the mapping from `c` to `t`.
  - We get e<sub>c</sub> from `E`. o<sub>c</sub>
  - A softmax layer then computes `P(t|c)`, which is y&#770;.
  - The cross-entropy loss function is used.
  - This is known as the skip-gram model.
- The softmax bottleneck in this model:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//39.png)
  - The denominator sums over all 10,000 vocabulary words.
  - With a vocabulary of 1 million or more, this computation becomes prohibitively slow.
- One remedy is the "**Hierarchical softmax classifier**", which structures the classification as a tree:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//40.png)
- In practice, the hierarchical softmax tree is not balanced. Frequent words sit near the root while rare words are placed at deeper levels.
- How to sample the context word **c**?
  - One option: pick uniformly at random from the corpus.
  - Problem: high-frequency words like "the, of, a, and, to, ..." would dominate, overshadowing words like "orange, apple, durian, ..."
  - In practice, heuristics are employed to balance frequent and infrequent words.
- The original Word2Vec paper presents two approaches to learning embeddings: the skip-gram model and CBoW (Continuous Bag of Words).

#### ➖ Negative Sampling Technique

- Negative sampling achieves similar objectives to the skip-gram model but with a far more efficient learning procedure. It reformulates the learning problem.
- Given the example sentence:
  
  - "I want a glass of orange juice to go along with my cereal"
- The sampling table looks like this:
- | Context | Word  | target |
  | ------- | ----- | ------ |
  | orange  | juice | 1      |
  | orange  | king  | 0      |
  | orange  | book  | 0      |
  | orange  | the   | 0      |
  | orange  | of    | 0      |

  The positive example is obtained via the skip-gram technique with a fixed window.
- Negative examples are drawn randomly from the vocabulary.
- Note that "of" appears as a negative example despite being present in the same sentence.
- Sample generation steps:
  1. Select a positive context-target pair.
  2. Draw k negative context-target pairs from the dictionary.
- Recommended k values: 5–20 for small datasets; 2–5 for larger ones.
- The training data has a ratio of k negatives to 1 positive per context word.
- Model definition for this supervised learning problem:
  - Let `c` be the context word, `t` the candidate word, and `y` the binary label.
  - A simple logistic regression model is applied.   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//41.png)
  - The logistic regression model can be visualized as:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//42.png)
  - Effectively, we have 10,000 binary classification sub-problems, but we only train k+1 of them per iteration.
- Strategies for selecting negative samples:
  - Sampling proportional to word frequency in the corpus seems natural, but it over-represents stop words like _the, of, and..._
  - The authors recommend this formula:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//43.png)

#### 🌍 GloVe Vectors

- GloVe is yet another word embedding algorithm — notable for its simplicity.
- It's less popular than Word2Vec and skip-gram models, though it has a dedicated following due to its straightforward design.
- GloVe stands for **Global Vectors for Word Representation**.
- Using the same running example: "I want a glass of orange juice to go along with my cereal".
- Select a context and target from the previously discussed options.
- Then compute for every pair: X<sub>ct</sub> = # times `t` appears in the context of `c`
- X<sub>ct</sub> = X<sub>tc</sub> when using a symmetric window (which GloVe does). They would differ with an asymmetric context window.
- The model is formulated as:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//44.png)
- f(x) — the weighting function — serves multiple purposes:
  - Avoids the `log(0)` problem when a context-target pair has zero co-occurrences.
  - Prevents stop words like "is", "the", and "this" from being over-weighted.
  - Ensures infrequent words aren't under-weighted.
- **Theta** and **e** are symmetric, which is exploited when computing the final word embeddings.

- _Takeaways on word embeddings:_
  - For your first attempt, download pre-trained embeddings — they typically work best.
  - If you have ample data, consider training your own using one of the algorithms above.
  - Word embedding training is computationally expensive, so most practitioners rely on pre-trained models.
  - One caveat: there's no guarantee that the learned feature axes will correspond to human-interpretable concepts like gender, royalty, or age.

### 🛠️ Practical Uses of Word Embeddings

#### 😊 Classifying Sentiment

- Sentiment classification determines whether text expresses a positive or negative opinion. It's a widely used NLP capability with applications across many domains. Example:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//45.png)
- A common challenge is the limited availability of large labeled datasets. Word embeddings can help compensate for this scarcity.
- Typical dataset sizes range from 10,000 to 100,000 labeled examples.
- A straightforward sentiment classification architecture:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//46.png)
  - The embedding matrix may have been trained on 100 billion words.
  - Each word embedding has 300 features.
  - We can **sum** or **average** all word embeddings then feed the result to a softmax classifier. This approach handles sentences of any length.
- A flaw of this simple model: it disregards word order. For instance, "Completely lacking in **good** taste, **good** service, and **good** ambience" contains "good" three times yet conveys a negative sentiment.
- An improved model uses an RNN:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//47.png)
  - Training this architecture yields a quite effective sentiment classifier.
  - It generalizes well even to words absent from the labeled training set. For example, if the sentence reads "Completely **<u>absent</u>** of good taste, good service, and good ambience" and "absent" never appeared in your labeled data — but it was present in the billion-word corpus used to train the embeddings — the model can still handle it correctly.

#### ⚖️ Removing Bias from Word Embeddings

- It's essential to ensure word embeddings don't encode harmful biases such as gender, ethnicity, or other prejudices.
- Troubling analogy results from trained embeddings:
  - Man : Computer_programmer as Woman : **Homemaker**
  - Father : Doctor as Mother : **Nurse**
- Word embeddings can mirror biases related to gender, ethnicity, age, sexual orientation, and more, all inherited from the training text.
- Since learning algorithms increasingly drive important decisions, bias-free models are critical.
- Andrew Ng believes we currently have more effective strategies for de-biasing AI than for de-biasing society, though significant work remains.
- Steps to mitigate bias in word embeddings:
  - Based on the paper: https://arxiv.org/abs/1607.06520
  - Starting from these learned embeddings:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//48.png)
  - We focus on **gender bias** here, but the approach generalizes to other bias types.
  - The procedure:
    1. Identify the bias direction:
       - Compute differences between embedding pairs:
         - e<sub>he</sub> - e<sub>she</sub>
         - e<sub>male</sub> - e<sub>female</sub>
         - ....
       - Average k such difference vectors.
       - This reveals the bias axis:   
         ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//49.png)
       - The result is a 1D bias direction and a 299D non-bias subspace.
    2. Neutralize: For words that aren't inherently gendered, project them onto the non-bias axis.
       - Words like babysitter and doctor should be gender-neutral, so we project them along the bias direction:   
         ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//50.png)
         - After neutralization, these words become equidistant from gendered words.
         - The authors trained a classifier to determine which words require neutralization.
    3. Equalize pairs:
       - Ensure gendered word pairs differ only along the gender dimension:
         - Grandfather – Grandmother
         - He – She
         - Boy – Girl
       - This is necessary because, before correction, the distance from "grandfather" to "babysitter" may differ from "grandmother" to "babysitter":   
         ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//51.png)
       - The fix is to move both members of each pair to be equidistant from the non-bias axis center.
       - The number of word pairs requiring this step is relatively small.


## 🎯 Sequence-to-Sequence Models and Attention

> Sequence models become significantly more powerful when augmented with attention mechanisms. This technique helps the model determine which parts of the input to focus on at each step. This section also covers speech recognition and working with audio data.

### 🏗️ Encoder-Decoder Architectures

#### 🧩 Foundational Models

- This section covers sequence-to-sequence (_Many to Many_) architectures, which are essential for tasks like machine translation and speech recognition.
- Starting with the basic setup:
  - A machine translation problem where X is a French sentence and Y is the English translation.   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//52.png)
  - The architecture has two components: an **encoder** and a **decoder**.
  - The encoder (an RNN — typically LSTM or GRU) processes the input sequence and produces a vector representation of the entire input.
  - The decoder (also an RNN) takes that representation and generates the output sequence.   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//53.png)
  - Key references:
    - [Sutskever et al., 2014. Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215)
    - [Cho et al., 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078)
- A similar architecture applies to image captioning:
  - Here X is an image and Y is the caption text.
  - Model diagram:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//54.png)
  - A pre-trained CNN (such as AlexNet) serves as the image encoder; the decoder is an RNN.
  - Related papers:
    - [Maoet et. al., 2014. Deep captioning with multimodal recurrent neural networks](https://arxiv.org/abs/1412.6632)
    - [Vinyals et. al., 2014. Show and tell: Neural image caption generator](https://arxiv.org/abs/1411.4555)
    - [Karpathy and Li, 2015. Deep visual-semantic alignments for generating image descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

#### 🎯 Selecting the Optimal Output Sentence

- The language model we learned earlier closely resembles the decoder portion of the translation model, with the key difference being a<sup>\<0></sup>   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//55.png)
- The problem formulations differ:
  - Language model: P(y<sup>\<1></sup>, ..., y<sup>\<Ty></sup>)
  - Machine translation: P(y<sup>\<1></sup>, ..., y<sup>\<Ty></sup> | x<sup>\<1></sup>, ..., x<sup>\<Tx></sup>)
- For machine translation, we don't want to sample the output randomly — that could yield suboptimal translations.
  - Example: 
    - X = "Jane visite l'Afrique en septembre."
    - Y might be:
      - Jane is visiting Africa in September.
      - Jane is going to be visiting Africa in September.
      - In September, Jane will visit Africa.
- The goal is to find the best possible output:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//56.png)
- The most widely used algorithm for this is beam search (detailed next).
- Why not use greedy search (always picking the highest-probability word at each step)?
  - It doesn't work well in practice.
  - Example: the ideal translation is "Jane is visiting Africa in September."
  - With greedy decoding, after generating "Jane is", the most probable next word might be "going" (since "going" commonly follows "<Noun> is"), producing: "Jane is going to be visiting Africa in September." — not the optimal result.
- Instead of greedy search, we need an approximate algorithm that maximizes the overall output probability.

#### 🔍 Beam Search Algorithm

- Beam search is the dominant algorithm for finding high-quality output sequences. It's a heuristic search method.
- Continuing with the translation example: we want Y = "Jane is visiting Africa in September."
- The algorithm uses a parameter `B` (beam width). With `B = 3`, the algorithm maintains 3 candidate sequences at each step.
- Step 1: identify the top B candidate first words — e.g., ["in", "jane", "september"].
- Step 2: for each first-word candidate, evaluate B possible second words and keep the top B two-word combinations by joint probability P(y<sup>\<1></sup>|x) * P(y<sup>\<2></sup>|x,y<sup>\<1></sup>). Result: ["in september", "jane is", "jane visit"]. Notice that "september" is automatically dropped as a first word.
- Continue this process for subsequent positions.
- Only B network instances are maintained at any point.
- Setting `B = 1` reduces beam search to greedy search.

#### 🔧 Improving Beam Search

- The basic beam search can be refined in several ways.
- **Length normalization**:
  - The beam search objective is to maximize:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//56.png)
  - Which involves multiplying:   
    P(y<sup>\<1></sup> | x) * P(y<sup>\<2></sup> | x, y<sup>\<1></sup>) * ... * P(y<sup>\<t></sup> | x, y<sup>\<y(t-1)></sup>)
  - Each probability is a fraction — usually a small one.
  - Multiplying many small fractions causes **numerical underflow** — values too small for floating-point precision.
  - The fix: use **log probabilities** (sum of logs instead of product of probabilities).   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//57.png)
  - However, both formulations favor shorter sequences (fewer terms = larger product/sum).
  - To counter this, normalize by sequence length:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//58.png)
    - Alpha is a tunable hyperparameter.
    - Alpha = 0 means no normalization.
    - Alpha = 1 means full normalization.
    - Alpha = 0.7 is a practical sweet spot.
- **Choosing beam width `B`**:
  - Larger B yields better results but increases computation.
  - Production systems often use `B=10`.
  - `B=100` or `B=1000` are rare (mostly research settings).
  - Unlike exact search algorithms such as BFS or DFS, beam search is faster but provides no guarantee of finding the globally optimal solution.

#### 🐛 Diagnosing Errors in Beam Search

- We previously covered **error analysis** in the _"Structuring Machine Learning Projects"_ course. Those principles apply here to debug beam search.
- The aim is to determine whether suboptimal translations stem from the beam search (insufficient `B`) or the RNN model itself.
- Example scenario:
  - Given information:
    - x = "Jane visite l'Afrique en septembre."
    - y<sup>*</sup> = "Jane visits Africa in September." — correct translation
    - y&#770; = "Jane visited Africa last September." — model output
  - The model produced a flawed result.
  - To assign blame, compute P(y<sup>*</sup> | X) and P(y&#770; | X). Two outcomes:
    - Case 1 (P(y<sup>*</sup> | X)  > P(y&#770; | X)): 
      - Verdict: Beam search is the culprit.
    - Case 2 (P(y<sup>*</sup> | X)  <= P(y&#770; | X)): 
      - Verdict: The RNN model is at fault.
- Error analysis procedure:
  - Select N error examples and build this table:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//59.png)
  - `B` marks beam search errors, `R` marks RNN errors.
  - Tally the counts to decide where to focus improvement efforts.

#### 📊 BLEU Score Evaluation

- A fundamental challenge in machine translation: any source sentence may have multiple valid translations. How do we score our outputs?
- The answer is the **BLEU score** (_Bilingual Evaluation Understudy_).
- The core idea: if the machine-generated translation closely resembles any human reference translation, it earns a high BLEU score.
- Worked example:
  - X = "Le chat est sur le tapis."
  - Y1 = "The cat is on the mat." (reference 1)
  - Y2 = "There is a cat on the mat." (reference 2)
  - Machine output: "the the the the the the the."
  - Naive evaluation: check if each output word appears in any reference. This gives _precision_:
    - precision = 7/7  (since "the" is found in Y1 or Y2)
  - This metric is clearly useless!
  - Better: **modified precision** — cap each word's count by its maximum occurrence in any single reference:
    - modified precision = 2/7 ("the" appears at most 2 times in Y1)
    - The raw count of 7 is clipped to the reference maximum of 2.
  - This evaluates one word at a time (unigrams). We can extend to n-grams.
- BLEU score using bigrams:
  - **N-grams** are contiguous subsequences of n items. An n-gram of size 1 is a "unigram"; size 2 is a "bigram"; size 3 is a "trigram".
  - X = "Le chat est sur le tapis."
  - Y1 = "The cat is on the mat."
  - Y2 = "There is a cat on the mat."
  - Machine output: "the cat the cat on the mat."
  - Bigrams in the machine output:
  
    | Pairs      | Count | Count clip |
    | ---------- | ----- | ---------- |
    | the cat    | 2     | 1 (Y1)     |
    | cat the    | 1     | 0          |
    | cat on     | 1     | 1 (Y2)     |
    | on the     | 1     | 1 (Y1)     |
    | the mat    | 1     | 1 (Y1)     |
    | **Totals** | 6     | 4          |

    Modified precision = sum(Count clip) / sum(Count) = 4/6
- Generalized modified precision for n-grams:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//60.png)
- Formalizing the complete BLEU score:
  - **P<sub>n</sub>** = BLEU score for a single n-gram type
  - **Combined BLEU score** = BP * exp(1/n * sum(P<sub>n</sub>))
    - For BLEU-4, compute P<sub>1</sub>, P<sub>2</sub>, P<sub>3</sub>, P<sub>4</sub>, average them, and exponentiate.
  - **BP** = **Brevity Penalty** — penalizes overly short outputs, which would otherwise score artificially high.   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//62.png)
- Open-source BLEU score implementations are readily available.
- BLEU is used across systems including machine translation and image captioning.

#### 💡 Intuition Behind Attention

- Up to now, our sequence-to-sequence models used a fixed-length encoder output. The _attention_ technique improves on this significantly.
- Attention has become one of the most impactful innovations in deep learning.
- The problem with long sequences:
  - Consider this model with its inputs and outputs:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//63.png)
  - The encoder must compress the entire input into a single vector, and the decoder must unpack it into the full translation.
  - A human translator wouldn't memorize an entire sentence before translating — they'd work on portions at a time.
  - Model performance degrades as sentence length increases.
  - The attention model mimics human behavior by focusing on relevant portions, substantially boosting accuracy on longer sequences:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//64.png)
    -  Blue line: standard model; green line: model with attention.
- This section provides a high-level intuition; full details follow in the next section.
- Originally developed for machine translation, attention has since been applied to computer vision, Neural Turing Machines, and numerous other architectures.
- The foundational paper:
  - [Bahdanau et. al., 2014. Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)
- Intuitive walkthrough:
  - Assume the encoder is a bidirectional RNN:
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//65.png)
  - The French sentence is fed to the encoder, which produces a representation of the input.
  - A separate decoder RNN generates the English output, starting with "Jane".
  - Attention weights determine which input words are most relevant for generating each output word. To produce "jane", the model attends to "jane", "visite", "l'Afrique":   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//66.png)
  - alpha<sup>\<1,1></sup>, alpha<sup>\<1,2></sup>, and alpha<sup>\<1,3></sup> are the attention weights used.
  - For every output word, a unique set of attention weights governs which input words receive focus:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//67.png)

#### 🔬 The Attention Mechanism in Detail

- Let's formalize the attention intuition into a concrete implementation.
- First, a bidirectional RNN (typically LSTM-based) encodes the French input:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//68.png)
- For simplicity, assume a<sup>\<t'></sup> includes both forward and backward activations at time step t'.
- A unidirectional decoder RNN generates the output, using a context vector `c` computed from the attention weights. These weights indicate how much each a<sup>\<t'></sup> should contribute to the current output:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//69.png)
- Attention weights for each sequence element must sum to 1:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//70.png)
- The context vector `c` is computed as:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//71.png)
- How attention weights are calculated:
  - alpha<sup>\<t, t'></sup> = the degree of attention y<sup>\<t></sup> should devote to a<sup>\<t'></sup>
    - For instance, generating the first output word uses alpha<sup>\<1,1></sup>, alpha<sup>\<1,2></sup>, alpha<sup>\<1,3></sup>
  - The raw attention scores are passed through a softmax so they sum to 1:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//72.png)
  - Computing e<sup>\<t, t'></sup> requires a small neural network (often just one hidden layer, since this computation occurs at every time step):   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//73.png)
    - s<sup>\<t-1></sup> is the decoder RNN's hidden state, and a<sup>\<t'></sup> is the encoder's bidirectional activation.
- A drawback of this algorithm: its time complexity is quadratic (O(T<sub>x</sub> × T<sub>y</sub>)).
- A useful way to interpret attention is by visualizing the attention weight matrix:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//74.png)

### 🎙️ Audio and Speech Processing

#### 🗣️ Automatic Speech Recognition

- One of the most exciting advances in sequence-to-sequence modeling has been the development of highly accurate speech recognition systems.
- Problem definition:
  - X: audio clip
  - Y: transcript
  - Visualizing an audio clip:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//75.png)
    - Horizontal axis represents time; vertical axis shows air pressure variation.
  - What really is an audio recording? A microphone records little variations in air pressure over time, and it is these little variations in air pressure that your ear perceives as sound. You can think of an audio recording is a long list of numbers measuring the little air pressure changes detected by the microphone. We will use audio sampled at 44100 Hz (or 44100 Hertz). This means the microphone gives us 44100 numbers per second. Thus, a 10 second audio clip is represented by 441000 numbers (= 10 * 44100).
  - Working with raw audio waveforms is quite challenging.
  - Even the human ear doesn't process raw waveforms directly — it decomposes sound into different frequencies.
  - A standard preprocessing step is to generate a spectrogram, which mimics this frequency decomposition:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//76.png)
    - Horizontal axis is time; vertical axis shows frequency. Color intensity reflects energy (loudness) at each frequency — analogous to what the human ear does.
  - A spectrogram is created by sliding a window over the raw signal and computing the dominant frequencies in each window via Fourier transformation.
  - Historically, speech recognition relied on _phonemes_ — hand-engineered basic sound units. Linguists believed that decomposing audio into phonemes was the optimal path to speech recognition.
  - End-to-end deep learning eliminated the need for phonemes, enabled by the availability of large audio datasets.
  - Research datasets typically contain 300–3,000 hours of audio; the best commercial systems now train on 100,000+ hours.
- Building an accurate speech recognition system using the attention model:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//77.png)
- The _CTC cost_ (**Connectionist Temporal Classification**) is another effective approach:
  - Example: Y = "the quick brown fox"
  - The RNN processes input and output sequences of equal length:   
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//78.png)
  - Note: shown as a unidirectional RNN here, but bidirectional RNNs are used in practice.
  - In speech recognition, input X is typically much longer than output Y.
    - 10 seconds at 100Hz yields X with shape (1000, ). These 10 seconds don't contain 1000 characters.
  - CTC allows outputs like:
    - `ttt_h_eee<SPC>___<SPC>qqq___` — encoding "the q"
    - The underscore `_` is a special "blank" character; `<SPC>` represents a space.
    - CTC rule: collapse consecutive duplicate characters that aren't separated by a blank.
  - So 19 characters in the target Y can be produced from a 1000-step output using CTC's blank mechanism.
  - Based on:
    - [Graves et al., 2006. Connectionist Temporal Classification: Labeling unsegmented sequence data with recurrent neural networks](https://dl.acm.org/citation.cfm?id=1143891)
    - These ideas also powered Baidu's DeepSpeech system.
- Combining attention models and CTC cost enables the construction of highly accurate speech recognition systems.

#### 🔔 Wake Word / Trigger Word Detection

- Deep learning has enabled devices that respond to spoken wake words. These are trigger word detection systems.
- Example: Amazon's Alexa responds when you say "Alexa, what time is it?" and provides an answer.
- Popular trigger word systems:  
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//79.png)
- The research landscape for trigger word detection is still evolving — no single universally accepted algorithm exists yet. Here's one workable approach.
- Building a trigger word detection model:
  - X: audio recording
  - X is preprocessed into spectrogram features:
    - X<sup>\<1></sup>, X<sup>\<2></sup>, ... , X<sup>\<t></sup>
  - Y: binary labels — 0 for non-trigger audio, 1 for the trigger word.
  - Example architecture:  
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//80.png)
    - Vertical lines in the audio mark moments immediately following the trigger word. The corresponding labels are 1.
  - A drawback: this produces a heavily imbalanced dataset with far more zeros than ones.
  - A practical workaround: output 1 for several consecutive time steps after the trigger word, rather than just a single time step, before reverting to 0.  
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//81.png)  
    ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//85.png)


## 📎 Supplementary Material

### 🔄 Machine Translation with Attention (Notebook Walkthrough)

- The model is implemented using Keras layers.
- The attention model architecture:   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//83.png)
  - There are two separate LSTMs in this model. Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through $T_x$ time steps; the post-attention LSTM goes through $T_y$ time steps. 
  - The post-attention LSTM passes $s^{\langle t \rangle}, c^{\langle t \rangle}$ from one time step to the next. In the lecture videos, we were using only a basic RNN for the post-activation sequence model, so the state captured by the RNN output activations $s^{\langle t\rangle}$. But since we are using an LSTM here, the LSTM has both the output activation $s^{\langle t\rangle}$ and the hidden cell state $c^{\langle t\rangle}$. However, unlike previous text generation examples (such as Dinosaurus in week 1), in this model the post-activation LSTM at time $t$ does will not take the specific generated $y^{\langle t-1 \rangle}$ as input; it only takes $s^{\langle t\rangle}$ and $c^{\langle t\rangle}$ as input. We have designed the model this way, because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date. 
- What one "Attention" step does to calculate the attention variables $\alpha^{\langle t, t' \rangle}$, which are used to compute the context variable $context^{\langle t \rangle}$ for each timestep in the output ($t=1, \ldots, T_y$).   
  ![](https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//84.png)
  - The diagram uses a `RepeatVector` node to copy $s^{\langle t-1 \rangle}$'s value $T_x$ times, and then `Concatenation` to concatenate $s^{\langle t-1 \rangle}$ and $a^{\langle t \rangle}$ to compute $e^{\langle t, t'}$, which is then passed through a softmax to compute $\alpha^{\langle t, t' \rangle}$. 
