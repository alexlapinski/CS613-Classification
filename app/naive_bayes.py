import util
from scipy.stats import norm
import operator
from metrics import BinaryClassifierMetric

def compute_posterior(models, data_class_probability, test_data):
    """
    Return a dictionary of data_class and probability of that data_class for the given test_data
    :param data_class_probability: probability for each class
    :param models: dictionary of (data_class, feature) => probability_density_function
    :param test_data: row of test_data
    :return: dictionary of data_class => probability
    """

    result = {}

    for data_class_name in models:

        probability = 1.0

        for feature_name in models[data_class_name]:
            feature_mean = models[data_class_name][feature_name]["mean"]
            feature_std = models[data_class_name][feature_name]["standard_deviation"]
            feature_probability = norm.pdf(test_data[feature_name], loc=feature_mean, scale=feature_std)
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

    spam_class_name = 1
    not_spam_class_name = 0

    # 2. Randomize the data.
    randomized_data = util.randomize_data(data)

    # 3. Split the data in for training and testing
    training_data, test_data = util.split_data(randomized_data, training_data_ratio)

    # 4. Standardize Training Data (except for class labels)
    training_features, training_data_target = util.split_features_target(training_data)
    std_training_features, mean, std = util.standardize_data(training_features)

    # 5. Divides the training data into two groups: Spam samples, Non-Spam samples.
    target_groups = training_data_target.groupby(training_data_target)

    total_training_size = float(len(training_data))

    print "Computing probability of priors"
    data_class_probability = {class_name: len(target_group) / total_training_size
                              for (class_name, target_group) in target_groups}

    # 6. Creates Normal models for each feature for each class.
    print "Creating normal models for each feature, for each class"
    models = {}
    for class_name, target_group in target_groups:
        models[class_name] = {}
        for feature_name in training_features.columns:
            dataset = std_training_features.loc[target_group.index][feature_name]
            feature_mean = dataset.mean()
            feature_std = dataset.std()
            models[class_name][feature_name] = {"mean":feature_mean, "standard_deviation": feature_std}

    # 7. Classify each testing sample using these models and choosing the class label based
    #    on which class probability is higher.
    print "Evaluating models for each test data point"
    test_features, test_targets = util.split_features_target(test_data)
    std_test_features, _, _ = util.standardize_data(test_features, mean, std)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in xrange(len(std_test_features)):
        probability_per_class = compute_posterior(models, data_class_probability, std_test_features.iloc[i])

        # Select the class label of the class with highest probability
        assigned_class = max(probability_per_class.iteritems(), key=operator.itemgetter(1))[0]
        expected_class = test_targets.iloc[i]

        # Tally up each of our counters for performance measurements
        if expected_class == spam_class_name:
            if assigned_class == spam_class_name:
                true_positives += 1
            else: # assigned_class == not_spam_class_name
                false_negatives += 1
        else: # expected_class == not_spam_class_name
            if assigned_class == not_spam_class_name:
                true_negatives += 1
            else: # assigned_class == spam_class_name
                false_positives += 1

    # 8. Computes the following statistics using the testing data results:
    metrics = BinaryClassifierMetric(true_positives, false_positives, true_negatives, false_negatives)

    return metrics
