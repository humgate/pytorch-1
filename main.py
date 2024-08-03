from linear_regression.model_operator import linear_regression
from bin_classification.model_operator import binary_classification
from multi_classification.model_operator import multi_classification
from multi_class_cnn.model_operator import multi_class_classification_non_cnn
from multi_class_cnn.cnn_model_train_operator import multi_class_train_cnn
from multi_class_cnn.cnn_model_pred_operator import multi_class_pred_cnn
from multi_class_cnn_custom_dataset.dataset_operator import load_dataset
from multi_class_cnn_custom_dataset.model_operator import multi_class_cnn_model_operator
from multi_class_vit.model_operator import multi_class_vit_model_operator

if __name__ == '__main__':
    #  linear_regression()
    #  binary_classification()
    #  multi_classification()
    #  multi_class_classification_non_cnn()
    #  multi_class_train_cnn()
    #  multi_class_pred_cnn()
    #  create_custom_dataset()
    #  multi_class_cnn_model_operator()
    multi_class_vit_model_operator()
