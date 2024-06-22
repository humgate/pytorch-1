from matplotlib import pyplot as plt

from util.helper_functions import plot_decision_boundary


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
            plt.scatter(test_data, predictions, c="r", s=4, label="Prediction")

        # Legend
        plt.legend(prop={"size": 14})
        plt.show()

    @staticmethod
    def plot_loss_curves(epoch_count,
                         loss_values,
                         test_loss_values
                         ):
        plt.plot(epoch_count, loss_values, label="Train loss")
        plt.plot(epoch_count, test_loss_values, label="Test loss")
        plt.title("Training and test loss curves ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_decision_boundary(model,
                               train_features,
                               train_labels,
                               test_features,
                               test_labels):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, train_features, train_labels)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, test_features, test_labels)

        plt.show()
