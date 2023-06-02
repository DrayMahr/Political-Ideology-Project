# Data Cleaning Procedures

## 1. Remove the test set Data

The data selected for the survey and for obtaining labels were removed from the original dataset.

## 2. Remove posts under some sub-topics

From the survey results, we can find that there are some sub-topics under which it is difficult to determine whether they belong to social or economic aspects, so we remove those posts under the sub-topics that the percentage of the total sample of the sub-topics that are marked as “*Not Sure/Both*” exceeds 40% in the survey results. These sub-topics are as follows.

- i**mmigration**, **infrastructure, transportation, cyber, environment, terrorism, healthcare, agriculture**

## 3. Selected appropriate authors

After constructing a topic model and labeling all data in the training set with topic labels, authors were filtered according to the number of posts they published under the social and economic topic categories, respectively. 

The specific filtering rules are as follows:

1. The number of posts by the author under both two topics should be at least 5.
2.  The ratio of the number of posts by authors under the two topics should be at least greater than 1:10. For those exceeding this ratio, the posts under the topic with the higher number are randomly under-sampled to 10 times the number of posts under the topic with the lower number. 
3. After completing the above steps, the posts of authors with a total number of posts above 200 are randomly split into multiple copies according to the topic as multiple dummy authors, ensuring that the total number of posts of each author does not exceed 200.