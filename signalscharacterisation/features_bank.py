from signalscharacterisation.features_implementations import FeaturesImplementations as FeImpl


def call_features_by_indexes(features_indexes, x, settings, normalise=0):
    '''
    This function call a set of features by their indexes.

    :param features_indexes: The index of features to calculate. The list of available features could be retrieved
    by calling get_features_list() function
    :param x: The input signal (channel by samples)
    :param settings: The settings required for that function as a dictionary.
    :param normalise: Whether to normalise the results. It is done per FEATURE independently across of its values,
    NOT per channel.
    :return: The dictionary including "final_values", "time" of calculations, "measures_names" calculated in that
    function, and the "function_name".
    '''

    length = features_indexes.shape
    return_list = [dict() for i in range(length[0])]
    features_list = get_features_list()
    for i in features_indexes:
        return_list[i] = call_feature_by_name(features_list[i], x, settings, normalise)

    return return_list

def get_features_list():
    """
    Returns the list of available features with this library.

    :return:
    """
    return FeImpl.get_features_list()

def call_feature_by_name(feature_name, x, settings, normalise=0):
    """
    This function calls a feature by its names as a string.

    :param feature_name: The name of the function to calculate the feature.
    :param x: The input signal (channel by samples)
    :param settings: The settings required for that function as a dictionary.
    :param normalise: Whether to normalise the results. It is done per FEATURE independently across of its values,
    NOT per channel.
    :return: The dictionary including "final_values", "time" of calculations, "measures_names" calculated in that
    function, and the "function_name".
    """
    feature = getattr(FeImpl, feature_name)
    settings["is_normalised"] = normalise
    return feature(x, settings)
