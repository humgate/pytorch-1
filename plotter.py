from matplotlib import pyplot as plt


class Plotter:
    @staticmethod
    def plot_predictions(train_data,
                         train_labels,
                         test_data,
                         test_labels,
                         predictions=None
                         ):
        """
        Plots training data, test data and compares predictions.
        """
        plt.figure(figsize=(10, 7))

        # Plot training data in blue
        plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

        # Plot test data in green
        plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

        # Plot predictions if they exists
        if predictions is not None:
            plt.scatter(test_data, predictions, c="r")

        # Legend
        plt.legend(prop={"size": 14})
        plt.show()
