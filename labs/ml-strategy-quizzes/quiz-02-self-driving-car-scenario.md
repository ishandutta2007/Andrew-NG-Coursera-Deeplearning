## Week 2 Quiz — Autonomous Vehicle Perception (Applied Case Study)

1. You're at the very beginning of this project. What should be your initial step? Assume that each of the actions listed below requires roughly the same amount of time (a few days).

    - Spend a few days training a basic model and see what mistakes it makes.
    
    > As covered in the lectures, applied ML is an intensely iterative discipline. Building a baseline model and performing error analysis (examining its mistakes) will guide you toward the most fruitful directions.
    
2. Your objective is to identify road signs (stop signs, pedestrian crossings, construction warnings) and traffic signals (red and green lights) within images. You want to determine which of these objects are present in each frame. You intend to employ a deep neural network with ReLU activations in its hidden layers.

    For the output layer, using softmax would be appropriate since this constitutes a multi-task learning scenario. Is that correct?
    
    - [ ] True
    - [x] False
    
    > Softmax would be suitable only if exactly one of the categories (stop sign, speed bump, pedestrian crossing, green light, red light) appeared in each image.
    
3. You're performing error analysis and tallying the types of mistakes the model makes. Which of the following datasets should you examine manually, inspecting one image at a time?

    - [ ] 10,000 images selected at random
    - [ ] 500 images selected at random
    - [x] 500 images on which the algorithm made a mistake
    - [ ] 10,000 images on which the algorithm made a mistake
    
4. After several weeks of data work, your team has assembled the following:

    - 100,000 labeled photographs captured by your car's front-facing camera.
    - 900,000 labeled road photographs sourced from the internet.
    
    Every image's label vector precisely indicates which road signs and traffic signals are present (including combinations). For instance, y(i) = [1 0 0 1 0] signifies the image contains a stop sign and a red traffic light.
    Since this is a multi-task learning problem, every element of your y(i) vectors must be labeled. If a particular example equals [0 ? 1 1 ?], the learning algorithm cannot utilize that sample. Is this true?

    - [ ] True
    - [x] False

    > As demonstrated in the multi-task learning lecture, you can formulate the cost function so that unlabeled entries do not affect the gradient.
    
5. The data distribution you ultimately care about consists of images from your car's forward camera, which differs from the distribution of images you downloaded from the web. How should you divide the dataset into train/dev/test partitions?

    - [ ] Combine all 100,000 car images with the 900,000 internet images. Shuffle the full set. Partition the resulting 1,000,000 images into 600,000 for training, 200,000 for dev, and 200,000 for test.
    - [ ] Combine all 100,000 car images with the 900,000 internet images. Shuffle the full set. Partition the resulting 1,000,000 images into 980,000 for training, 10,000 for dev, and 10,000 for test.
    - [x] Choose the training set to be the 900,000 images from the internet along with 80,000 images from your car's front-facing camera. The 20,000 remaining images will be split equally in dev and test sets.
    - [ ] Choose the training set to be the 900,000 images from the internet along with 20,000 images from your car's front-facing camera. The 80,000 remaining images will be split equally in dev and test sets.
    > As explained in the lectures, your dev and test sets should mirror the distribution of "real-world" data as closely as possible. It's equally important that the training set includes sufficient "real" examples to prevent a data-mismatch issue.
    
6. Suppose you've settled on the following data partitioning:

    - Training	940,000 images randomly picked from (900,000 internet images + 60,000 car's front-facing camera images)	8.8%
    - Training-Dev	20,000 images randomly picked from (900,000 internet images + 60,000 car's front-facing camera images)	9.1%
    - Dev	20,000 images from your car's front-facing camera	14.3%
    - Test	20,000 images from the car's front-facing camera	14.8%
    
    You also know that human-level error for road sign and traffic signal classification is approximately 0.5%. Which of the following statements hold true? (Select all that apply.)
    
    - You have a large avoidable-bias problem because your training error is quite a bit higher than the human-level error.
    - You have a large data-mismatch problem because your model does a lot better on the training-dev set than on the dev set.
    
7. Referring to the table in the previous question, a colleague hypothesizes that the training data distribution is significantly easier than the dev/test distribution. What is your assessment?

    - There's insufficient information to tell if your friend is right or wrong.
    
    > The model performs better on data drawn from the distribution it was trained on. However, you cannot determine whether this is because it was specifically trained on that distribution or because it is inherently easier. To gain clarity, evaluate human-level error independently on both distributions.
    
8. You choose to focus on the dev set and manually categorize the sources of error. Here is a summary of your findings:

    - Overall dev set error	14.3%
    - Errors due to incorrectly labeled data	4.1%
    - Errors due to foggy pictures	8.0%
    - Errors due to rain drops stuck on your car's front-facing camera	2.2%
    - Errors due to other causes	1.0%
    
    In this breakdown, the percentages (4.1%, 8.0%, etc.) represent fractions of the entire dev set — not just misclassified examples. Put differently, roughly 8.0/14.3 ≈ 56% of all errors stem from foggy conditions.

    This analysis means the team's top priority should be acquiring more foggy images for the training set to tackle the 8.0% error contribution from that category. True or False?
    
    - [x] False because this would depend on how easy it is to add this data and how much you think your team thinks it'll help.
    - [ ] True because it is the largest category of errors. As discussed in lecture, we should prioritize the largest category of error to avoid wasting the team's time.
    - [ ] True because it is greater than the other error categories added together (8.0 > 4.1+2.2+1.0).
    - [ ] False because data augmentation (synthesizing foggy images by clean/non-foggy images) is more efficient.
    
9. You have the option to purchase a specially engineered windshield wiper that can clear some of the raindrops from the forward-facing camera. Based on the error breakdown in the preceding question, which statement do you find most reasonable?

    - 2.2% would be a reasonable estimate of the maximum amount this windshield wiper could improve performance.
    
    > Your performance gain is unlikely to exceed 2.2% from resolving the raindrop issue. With an infinitely large dataset, 2.2% would precisely quantify the improvement achievable by installing a specialized raindrop-clearing windshield wiper.
    
10. You opt to use data augmentation to tackle the foggy-image problem. You locate 1,000 fog photographs online and overlay them onto clean images to simulate foggy conditions, as follows:

    Which of the following statements do you consider valid? (Select all that apply.)
    
    - So long as the synthesized fog looks realistic to the human eye, you can be confident that the synthesized data is accurately capturing the distribution of real foggy images, since human vision is very accurate for the problem you're solving.
    
    > If the synthetically fogged images appear genuine, the model will treat them as additional useful training examples for recognizing road signs and traffic signals in hazy weather. This approach is very likely to be beneficial.
    
11. After continued work on the problem, you've chosen to fix the mislabeled examples in the dev set. Which of these statements do you consider correct? (Select all that apply.)

    - You should not correct incorrectly labeled data in the training set as well so as to avoid your training set now being even more different from your dev set.
    
    >  Deep learning algorithms are quite robust to having slightly different train and dev distributions. 
    
    - You should also correct the incorrectly labeled data in the test set, so that the dev and test sets continue to come from the same distribution
    
    > Maintaining identical distributions between dev and test sets ensures your team's iterative development loop remains efficient and reliable.
    
12. Currently your algorithm only detects red and green traffic lights. A colleague at your startup has begun working on recognizing yellow traffic lights. (In some countries this is called an amber or orange light; we'll follow the US convention and call it yellow.) Images with yellow lights are quite scarce, and she lacks sufficient data to train a robust model. She's hoping you can assist via transfer learning.

    What guidance do you offer?
    
    - She should try using weights pre-trained on your dataset, and fine-tuning further with the yellow-light dataset.
    
    > Your model has been trained on a massive dataset, while her dataset is small. Although the label sets differ, your model's parameters have learned to recognize many visual characteristics of roads and traffic scenes that are directly relevant to her task. This is an ideal scenario for transfer learning — she can adopt your architecture, replace the final output layer, and initialize with your pre-trained weights.

13. A different colleague proposes using externally mounted microphones to detect nearby vehicles by sound — for instance, recognizing a police siren approaching from behind. However, they have very little training data for this audio system. How can you contribute?

    - Neither transfer learning nor multi-task learning seems promising.
    
    > The audio detection problem is fundamentally different from your visual recognition task. The distinct data modalities and structures make both transfer learning and multi-task learning impractical here.
    
14. To identify red and green lights, you have been following this methodology:

    - (A) Feed an image (x) into a neural network and have it learn a direct mapping to predict whether a red light and/or green light is present (y).
    
    A teammate suggests an alternative two-stage pipeline:
    
    - (B) First (i) localize the traffic light within the image (if one exists), then (ii) classify the color of the illuminated lamp.
    Between these two approaches, Approach B is more end-to-end because it features separate stages for the input and output ends. True or False?


    - [ ] True
    - [x] False
    
    >  (A) is an end-to-end approach as it maps directly the input (x) to the output (y).
    
15. Approach A (described above) tends to outperform Approach B when you have a ________ (fill in the blank).

    - [x] Large training set
    - [ ] Multi-task learning problem.
    - [ ] Large bias problem.
    - [ ] Problem with a high Bayes error.

    > Across many domains, end-to-end learning has been shown to perform well in practice, but it demands a substantial volume of training data.
