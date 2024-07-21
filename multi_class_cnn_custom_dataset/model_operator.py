from multi_class_cnn_custom_dataset.dataset_operator import load_dataset


def multi_class_cnn_model_operator():
    train_dataloader, test_dataloader = load_dataset(
        "data/pizza_steak_sushi/",
        "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        32
    )




