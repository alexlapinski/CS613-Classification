import util
import math
from functools import partial


def probability_density(x, mean, standard_deviation):
    """
    Compute the probability density of value x given the mean and standard deviation.
    :param x: Observed value
    :param mean: Mean of the feature space
    :param standard_deviation: Standard deviation of the feature space
    :return: Probability
    """

    coefficient = 1.0 / standard_deviation * math.sqrt(2 * math.pi)
    exponent = (-1.0 / 2.0) * ((x - mean) / standard_deviation)**2

    return coefficient * math.e ** exponent


def execute(data, training_data_ratio=2.0/3):
    """
    Execute the Naive Bayes classification
    :param data: Dataframe containing training and test data
    :param training_data_ratio:
    :return:
    """

    print data.columns
    print data.head()

    # 2. Randomizes the data.
    randomized_data = util.randomize_data(data)

    print randomized_data.head()

    # 3. Selects the first 2 / 3(round up) of the data for training and the remaining for testing
    training_data, test_data = util.split_data(randomized_data, training_data_ratio)

    # 4. Standardizes the data(except for the last column of course) using the training data
    std_training_data, mean, std = util.standardize_data(training_data)

    # 5. Divides the training data into two groups: Spam samples, Non-Spam samples.
    spam_training_data = std_training_data.loc[1]
    not_spam_training_data = std_training_data.loc[0]

    # 6. Creates Normal models for each feature for each class.
    normal_models = {}
    for feature in data.columns:
        feature_mean = std_training_data[feature].mean()
        feature_std = std_training_data[feature].std()
        normal_models[feature] = partial(probability_density, mean=feature_mean, standard_deviation=feature_std)

    # 7. Classify each testing sample using these models and choosing the class label based
    #    on which class probability is higher.


    # 8. Computes the following statistics using the testing data results:
    #   (a) Precision
    #   (b) Recall
    #   (c) F - measure
    #   (d) Accuracy


    # 2. If you decide to work in log space, realize that Matlab interprets 0log0 as NaN (not a number).
    #    You should identify this situation and consider it to be a value of zero.

    # 3. Although Naive Bayes Classiers can do multi-class classication directly, you may assume
    #    binary classication in your implementation.

    pass