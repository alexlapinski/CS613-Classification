from sklearn.svm import SVC
import util
import random
from itertools import combinations
from metrics import AccuracyMetric


def execute(dataframe, training_data_ratio=2.0/3):
    """
    Execute Multi-class SVM
    :param dataframe: The input dataset containing the classifier as the last column
    :param training_data_ratio: The percentage of data to use for training (default: 2/3)
    :return: A list of metrics on performance for the one-vs-many, and the accuracy of one-vs-one SVM
    """

    # Seed our randomizer to ensure we get repeatable results
    random.seed(0)

    # 2. Randomizes the data.
    randomized_data = util.randomize_data(dataframe)

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining for testing
    training_data, test_data = util.split_data(randomized_data, training_data_ratio)

    # 4. Standardizes the data (except for the last column of course) using the training data
    training_features, training_targets = util.split_features_target(training_data)
    std_training_features, mean, std = util.standardize_data(training_features)

    # Due to the standard deviation being zero, we end up with NaN entries, reset them to zero
    std_training_features.fillna(0, inplace=True)

    test_features, test_targets = util.split_features_target(test_data)
    std_test_features, _, _ = util.standardize_data(test_features, mean, std)

    # Due to the standard deviation being zero, we end up with NaN entries, reset them to zero
    std_test_features.fillna(0, inplace=True)

    target_classes = training_targets.unique()

    # 5. First trains and evaluates using a One vs All approach:
    clf = SVC()
    one_vs_many_metrics = []
    print "Training/Testing one-vs-many SVM"
    for target_class in target_classes:
        others = [cls for cls in target_classes if cls != target_class]
        print "{0} vs {1}".format(target_class, others)

        # (a) Train K binary classifiers, one per class. When training classifier i, set all instances
        #     pertaining to class i's label to one, and set the label of all other instances to zero.
        temp_targets = training_targets.apply(lambda x: 1.0 if x == target_class else 0.0)
        clf.fit(std_training_features, temp_targets)

        # (b) For each test sample, run it through each of the K classifiers to see which class(es) it
        #     belongs to vs the "others". In other words, if when testing a sample using classifier i you
        #     get back a label of one, then this sample was thought to belong to class i. If you got back
        #     zero, then it was not. If there's multiple classes it belongs to, select one of those classes
        #     at random as the predicted class label.
        num_classified_incorrectly = 0
        for i in xrange(len(std_test_features)):
            actual_class = clf.predict(std_test_features.iloc[i].reshape(1, -1))
            if actual_class == 1:
                actual_class = target_class
            else:
                actual_class = random.choice(others)
            expected_class = test_targets.iloc[i]
            if expected_class != actual_class:
                num_classified_incorrectly += 1

        # (c) Since there's more than one class, the concept of "Positive" and "Negative" don't really apply.
        #     Therefore just compute the accuracy as the percentage of samples classified correctly
        metric = AccuracyMetric(len(std_test_features), num_classified_incorrectly, target_class, others)
        one_vs_many_metrics.append(metric)

    # 6. Trains and evaluates using a One vs One approach:

    # Construct our possible 1vs1 sets
    one_vs_one_models = {}
    print "Training one-vs-one SVM"
    for binary_classes in combinations(target_classes, 2):
        print "{0} vs {1}".format(binary_classes[0], binary_classes[1])
        # (a) Train K(K-1)/2 one-vs-one binary classifiers where you only use the training samples from
        #     the relevant classes. That is, if you are training a classifier that labels observations as
        #     either class i or class k, then only use observations with labels i and j (discard the rest).
        input_targets = training_targets[training_targets.isin(binary_classes)]
        input_features = std_training_features.ix[input_targets.index]
        one_vs_one_models[binary_classes] = SVC()
        one_vs_one_models[binary_classes].fit(input_features, input_targets)

    num_classified_incorrectly = 0
    print "Testing one-vs-one for each test sample"
    for i in xrange(len(std_test_features)):
        class_assignments = {class_name: 0 for class_name in target_classes}
        row_features = std_test_features.iloc[i]

        for binary_classes, model in one_vs_one_models.items():
            # (b) For each test sample, run it through each of the K(K - 1)/2 classifiers to see which
            #     class(es) "beat" the others the most. Choose that class as the your observation's label.
            #     Again if there is a tie among several classes, choose at random the predicted label from
            #     those classes.
            result_class = model.predict(row_features.reshape(1, -1))
            class_assignments[result_class[0]] += 1

        # Get the class_name which won the most
        actual_class = max(class_assignments, key=class_assignments.get)

        # Check to see if there are multiple classes which tied
        tied_classes = [key for key, val in class_assignments.items() if val == class_assignments[actual_class]]
        if len(tied_classes) > 1:
            actual_class = random.choice(tied_classes)

        expected_class = test_targets.iloc[i]

        # (c) Since there's more than one class, the concept of "Positive" and "Negative" don't really apply.
        #     Therefore just compute the accuracy as the percentage of samples classified correctly confident.
        if actual_class != expected_class:
            num_classified_incorrectly += 1

    num_classified_correctly = len(test_features) - num_classified_incorrectly
    one_vs_one_accuracy = num_classified_correctly / float(len(test_features))

    return one_vs_many_metrics, one_vs_one_accuracy
