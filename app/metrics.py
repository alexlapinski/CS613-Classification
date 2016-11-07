class BinaryClassifierMetric:

    def __init__(self, true_positives, false_positives, true_negatives, false_negatives):
        """
        Create a new Metrics object to record performance of a classification algorithm

        :param true_positives: Number of true positives
        :param false_positives: Number of false positives
        :param true_negatives: Number of true negatives
        :param false_negatives: Number of false negatives
        """

        assert true_positives >= 0, "True Positives must be greater than or equal to zero"
        assert false_positives >= 0, "False Positives must be greater than or equal to zero"
        assert true_negatives >= 0, "True Negatives must be greater than or equal to zero"
        assert false_negatives >= 0, "False Negatives must be greater than or equal to zero"

        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives

    def precision(self):
        """
        Compute the percentage of items classified as positive and were actually positive.
        :return: Percentage of items classified as positive and were actually positive.
        """
        return float(self.true_positives) / (self.true_positives + self.false_positives)

    def recall(self):
        """
        Compute the percentage of true positives correctly identified (aka True Positive Rate).
        :return: Percentage of true positives correctly identified.
        """
        return float(self.true_positives) / (self.true_positives + self.true_negatives)

    def false_positive_rate(self):
        """
        Compute the percentage of false positives over false positives and true negatives.
        :return: Percentage of false positives over false positives and true negatives.
        """
        return float(self.false_positives) / (self.false_positives + self.true_negatives)

    def f_measure(self):
        """
        Compute the weighted harmonic mean of precision and recall.
        :return: Weighted harmonic mean of precision and recall.
        """
        precision = float(self.precision())
        recall = float(self.recall())
        return (2 * precision * recall) / (precision + recall)

    def accuracy(self):
        """
        Compute the accuracy using all available metrics.
        :return: Percentage of correctly identified items over all identified items.
        """

        correctly_identified_items = (self.true_positives + self.true_negatives)
        incorrectly_identified_items = (self.false_positives + self.false_negatives)
        total_items = correctly_identified_items + incorrectly_identified_items

        return correctly_identified_items / float(total_items)


class AccuracyMetric:

    def __init__(self, total_samples, num_incorrectly_classified, one_class, other_classes):
        """
        Create a metric object to represent one-vs-many multi-classifier.
        :param total_samples: Total number of samples in the test population.
        :param num_incorrectly_classified: Number of samples incorrectly classified.
        :param one_class: Label for the 'one' class.
        :param other_classes: Labels for the 'other' classes, may be one or many
        """

        assert total_samples > 0, "Expected total samples to be greater than zero."
        assert num_incorrectly_classified <= total_samples, "Expected the number of incorrectly classified " \
                                                            "samples to be less than or equal to the total samples"

        assert one_class is not None, "Expected 'one_class' to be a non-None value."
        assert other_classes is not None, "Expected 'other_classes' to be a non-None value."

        self._total_samples = total_samples
        self._num_incorrectly_classified = num_incorrectly_classified
        self._one_class = one_class
        self._other_classes = other_classes

    @property
    def one_class(self):
        return self._one_class

    @property
    def other_classes(self):
        return self._other_classes

    def accuracy(self):
        """
        Compute the accuracy of this test run.
        :return: Percentage of samples correctly classified of all samples.
        """
        num_correctly_classified = self._total_samples - self._num_incorrectly_classified
        return num_correctly_classified / float(self._total_samples)