

def execute(data):
    """

    :param data:
    :return:
    """

    # 1. Reads in the data from the Cardiotocography set provided on Blackboard.
    #    Read about how we'll use this dataset in the Datasets section.

    # 2. Randomizes the data.

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining for testing

    # 4. Standardizes the data (except for the last column of course) using the training data

    # 5. First trains and evaluates using a One vs All approach:
    #    (a) Train K binary classifiers, one per class. When training classifier i, set all instances
    #        pertaining to class i's label to one, and set the label of all other instances to zero.
    #    (b) For each test sample, run it through each of the K classifiers to see which class(es) it
    #        belongs to vs the "others". In other words, if when testing a sample using classifier i you
    #        get back a label of one, then this sample was thought to belong to class i. If you got back
    #        zero, then it was not. If there's multiple classes it belongs to, select one of those classes
    #        at random as the predicted class label.
    #   (c) Since there's more than one class, the concept of "Positive" and "Negative" don't really apply.
    #       Therefore just compute the accuracy as the percentage of samples classified correctly

    # 6. Trains and evaluates using a One vs One approach:
    #   (a) Train K(K-1)/2 one-vs-one binary classifiers where you only use the training samples from
    #       the relevant classes. That is, if you are training a classifier that labels observations as
    #       either class i or class k, then only use observations with labels i and j (discard the rest).
    #   (b) For each test sample, run it through each of the K(K - 1)/2 classifiers to see which
    #       class(es) "beat" the others the most. Choose that class as the your observation's label.
    #       Again if there is a tie among several classes, choose at random the predicted label from
    #       those classes.
    #   (c) Since there's more than one class, the concept of "Positive" and "Negative" don't really apply.
    #       Therefore just compute the accuracy as the percentage of samples classified correctly confident.

    pass
