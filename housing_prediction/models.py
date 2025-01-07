"""Build, save and load models for the pipeline."""


def build_model(model_config=None):
    """Build a model with the specified configuration

    Args:
        model_config (dict[str, Any]): Model configuration.

    Returns:
        object: Model object.
    """
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(
        min_samples_leaf=10,
        random_state=42,
    )


def save_model(model, path):
    """Save the model to the specified path.

    Args:
        model (object): Model object to save.
        path (Path): Path to save the model.
    """
    import joblib

    joblib.dump(model, path)


def load_model(path):
    """Load the model from the specified path.

    Args:
        path (Path): Path to load the model from.

    Returns:
        object: Model object.
    """
    import joblib

    return joblib.load(path)
