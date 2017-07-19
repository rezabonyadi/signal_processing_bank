import signalscharacterisation.FeaturesImplementations as FeImpl


class SignalsFeatures:
    @staticmethod
    def call_features_by_indexes(features_indexes, x, settings):
        """
        This function call a set of features by their indexes.

        :param features_indexes: The index of features to calculate. The list of available features could be retrieved
        by calling get_features_list() function
        :param x: The input signal (channel by samples)
        :param settings: The settings required for that function as a dictionary.
        :return: The dictionary including "final_values", "time" of calculations, "measures_names" calculated in that
        function, and the "function_name".
        """

        length = features_indexes.shape
        return_list = [dict() for i in range(length[0])]
        features_list = SignalsFeatures.get_features_list()
        for i in features_indexes:
            return_list[i] = SignalsFeatures.call_feature_by_name(features_list[i], x, settings)

        return return_list

    @staticmethod
    def get_features_list():
        """
        Returns the list of available features with this library.

        :return:
        """
        return FeImpl.FeaturesImplementations.get_features_list()

    @staticmethod
    def call_feature_by_name(feature_name, x, settings):
        """
        This function calls a feature by its names as a string.

        :param feature_name: The name of the function to calculate the feature.
        :param x: The input signal (channel by samples)
        :param settings: The settings required for that function as a dictionary.
        :return: The dictionary including "final_values", "time" of calculations, "measures_names" calculated in that
        function, and the "function_name".
        """
        feature = getattr(FeImpl.FeaturesImplementations, feature_name)
        return feature(x, settings)
