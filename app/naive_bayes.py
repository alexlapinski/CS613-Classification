import util
from scipy.stats import norm
import operator


def compute_posterior(models, data_class_probability, test_data):
    """
    Return a dictionary of data_class and probability of that data_class for the given test_data
    :param data_class_probability: probability for each class
    :param models: dictionary of (data_class, feature) => probability_density_function
    :param test_data: row of test_data
    :return: dictionary of data_class => probability
    """

    #print "Computing Posterior for {0}".format(test_data)

    result = {}

    for data_class_name in models:

        probability = 1.0

        for feature_name in models[data_class_name]:
            feature_mean = models[data_class_name][feature_name]["mean"]
            feature_std = models[data_class_name][feature_name]["standard_deviation"]
            #print "Feature: {0}, Mean: {1}, Std: {2}".format(feature_name, feature_mean, feature_std)
            feature_probability = norm.pdf(test_data[feature_name], loc=feature_mean, scale=feature_std)
            #print "{0} Probability = {1}".format(feature_name, feature_probability)
            probability *= feature_probability

        result[data_class_name] = data_class_probability[data_class_name] * probability

    normalize_denominator = sum(result.itervalues())
    return {key: value / normalize_denominator for key, value in result.items()}


def execute(data, training_data_ratio=2.0 / 3):
    """
    Execute the Naive Bayes classification
    :param data: Dataframe containing training and test data
    :param training_data_ratio:
    :return:
    """

    # 2. Randomize the data.
    randomized_data = util.randomize_data(data)

    # 3. Split the data in for training and testing
    training_data, test_data = util.split_data(randomized_data, training_data_ratio)

    # 4. Standardize Training Data (except for class labels)
    std_training_data, mean, std = util.standardize_data(training_data)

    # 5. Divides the training data into two groups: Spam samples, Non-Spam samples.
    spam_training_data = std_training_data.loc[1]
    not_spam_training_data = std_training_data.loc[0]

    total_training_size = float(len(training_data))
    data_class_probability = {1: len(spam_training_data) / total_training_size,
                              0: len(not_spam_training_data) / total_training_size}

    print "Data Class Probability: {0}".format(data_class_probability)

    # 6. Creates Normal models for each feature for each class.
    models = {}
    for data_class in data.index.unique():
        models[data_class] = {}
        for feature in data.columns:
            feature_mean = std_training_data[feature].mean()
            feature_std = std_training_data[feature].std()
            models[data_class][feature] = {"mean":feature_mean, "standard_deviation": feature_std}

    # 7. Classify each testing sample using these models and choosing the class label based
    #    on which class probability is higher.
    std_test_data, _, _ = util.standardize_data(test_data, mean, std)
    for i in xrange(len(std_test_data)):
        probability = compute_posterior(models, data_class_probability, std_test_data.iloc[i])
        assigned_class = max(probability.iteritems(), key=operator.itemgetter(1))[0]
        if assigned_class == 0:
            continue
        print probability
        print assigned_class


    # 8. Computes the following statistics using the testing data results:
    #   (a) Precision
    #   (b) Recall
    #   (c) F - measure
    #   (d) Accuracy

    # 2. If you decide to work in log space, realize that Matlab interprets 0log0 as NaN (not a number).
    #    You should identify this situation and consider it to be a value of zero.

    # 3. Although Naive Bayes Classifiers can do multi-class classification directly, you may assume
    #    binary classification in your implementation.

    pass
