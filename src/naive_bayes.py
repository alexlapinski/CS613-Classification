

def execute(data):
    """

    :param data:
    :return:
    """

    # 1. Reads in the data.
    # 2. Randomizes the data.
    # 3. Selects the first 2 / 3(round up) of the data for training and the remaining for testing
    # 4. Standardizes the data(except for the last column of course) using the training data
    # 5. Divides the training data into two groups: Spam samples, Non - Spam samples.
    # 6. Creates Normal models for each feature for each class.
    # 7. Classify each testing sample using these models and choosing the class label based
    #    on which class probability is higher.
    # 8. Computes the following statistics using the testing data results:
    #   (a) Precision
    #   (b) Recall
    #   (c) F - measure
    #   (d) Accuracy

    # 1. Seed the random number generate with zero prior to randomizing the data
    # 2. If you decide to work in log space, realize that Matlab interprets 0log0 as NaN (not a number).
    #    You should identify this situation and consider it to be a value of zero.
    # 3. Although Naive Bayes Classiers can do multi-class classication directly, you may assume
    #    binary classication in your implementation.

    pass