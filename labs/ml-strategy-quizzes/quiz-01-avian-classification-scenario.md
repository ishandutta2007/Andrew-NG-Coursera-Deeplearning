## Week 1 Quiz — Avian Detection in Peacetopia (Applied Case Study)

1. When you have three separate evaluation criteria, it becomes more difficult to rapidly compare two candidate algorithms, which slows down your team's iteration cycle. Is this statement accurate?

    - [x] True
    - [ ] False

2. Given the three models below, which would you select?

    - Test Accuracy	98% 
    - Runtime 9 sec	
    - Memory size 9MB

3. Considering the municipality's requirements, which of the following descriptions is correct?

    - [x] Accuracy is an optimizing metric; running time and memory size are a satisficing metrics.
    - [ ] Accuracy is a satisficing metric; running time and memory size are an optimizing metric.
    - [ ] Accuracy, running time and memory size are all optimizing metrics because you want to do well on all three.
    - [ ] Accuracy, running time and memory size are all satisficing metrics because you have to do sufficiently well on all three for your system to be acceptable.

4. Prior to building your algorithm, you must partition your data into train/dev/test splits. Which partition scheme is most appropriate?

    - Train 9,500,000		
    - Dev 250,000
    - Test 250,000

5. Once your train/dev/test splits are established, the City Council obtains an additional 1,000,000 photographs referred to as "citizens' data." The residents of Peacetopia are so alarmed by birds that they took sky photographs and labeled them voluntarily, producing this extra 1,000,000 image set. These photographs follow a different distribution than the images originally supplied by the City Council, but you believe they could benefit your model.

	  It would be a mistake to incorporate the citizens' data into the training set, since doing so would make the training distribution diverge from the dev/test distribution, thereby degrading dev and test set accuracy. Is this correct?

    - [ ] True
    - [x] False

6. A City Council member with some ML knowledge suggests adding the 1,000,000 citizens' photographs to the test set. You push back because:

    - The test set no longer reflects the distribution of data (security cameras) you most care about.
    - This would cause the dev and test set distributions to become different. This is a bad idea because you're not aiming where you want to hit.

7. You build a system and observe the following error rates (error = 100% − Accuracy):
	
    - Training set error	4.0%
    - Dev set error	4.5%

    This indicates that a productive path to better results is training a larger network to reduce the 4.0% training error. Do you concur?

    - No, because there is insufficient information to tell.

8. You recruit several annotators to label the data and establish human-level accuracy. Their results are:

    - Birding specialist #1	0.3% error
    - Birding specialist #2	0.5% error
    - Non-expert individual #1	1.0% error
    - Non-expert individual #2	1.2% error

    If you want "human-level performance" to serve as an approximation for Bayes error, what should you use as "human-level performance"?

    - 0.3% (accuracy of expert #1)

9. Which of the following claims do you find valid?

	  - A learning algorithm's performance can be better human-level performance but it can never be better than Bayes error.

10. You discover that a panel of ornithologists collaboratively examining an image achieves an even stronger 0.1% error rate, so you adopt that as "human-level performance." After further development, your numbers look like this:

    - Human-level performance	0.1%
    - Training set error	2.0%
    - Dev set error	2.1%

    Given this evidence, which two of the following four approaches appear most worthwhile? (Select two.)

    - Try decreasing regularization.
    - Train a bigger model to try to do better on the training set.

11. You additionally measure your model on the test set and observe:

    - Human-level performance	0.1%
    - Training set error	2.0%
    - Dev set error	2.1%
    - Test set error	7.0%

    How should you interpret these results? (Pick the two best explanations.)

    - You should try to get a bigger dev set.
    - You have overfit to the dev set.

12. After a full year of effort on this project, you finally reach:

    - Human-level performance	0.10%
    - Training set error	0.05%
    - Dev set error	0.05%

    What conclusions can you draw? (Select all that apply.)

    - It is now harder to measure avoidable bias, thus progress will be slower going forward.
	  - If the test set is big enough for the 0,05% error estimate to be accurate, this implies Bayes error is ≤0.05

13. It turns out Peacetopia engaged one of your rivals to build a competing system. Both your system and your rival's have comparable latency and memory footprint. Yet your system boasts superior overall accuracy! Nonetheless, when Peacetopia evaluates both systems in practice, they prefer your rival's, because despite your higher aggregate accuracy, your system produces more false negatives (it fails to sound the alarm when a bird is actually airborne). What is the right course of action?

	  - Rethink the appropriate metric for this task, and ask your team to tune to the new metric.

14. You've decisively outperformed your competitor, and your system is now deployed across Peacetopia, safeguarding its citizens from birds! However, over recent months a previously unseen bird species has been gradually migrating into the region, causing your system's accuracy to decline as it encounters data it has never been trained on.

	  - Use the data you have to define a new evaluation metric (using a new dev/test set) taking into account the new species, and use that to drive further progress for your team.

15. The City Council believes that introducing more cats into the city could deter the birds. They're so pleased with your bird detector that they've also commissioned you to develop a cat detector. (Evidently, cat detectors are extraordinarily versatile.) Thanks to years of prior work on cat recognition, you possess a massive dataset of 100,000,000 cat images, and training on this entire collection takes roughly two weeks. Which of the following do you agree with? (Mark all that apply.)

    - If 100,000,000 examples is enough to build a good enough Cat detector, you might be better of training with just 10,000,000 examples to gain a ≈10x improvement in how quickly you can run experiments, even if each model performs a bit worse because it's trained on less data.
    - Buying faster computers could speed up your teams' iteration speed and thus your team's productivity.
    - Needing two weeks to train will limit the speed at which you can iterate.
