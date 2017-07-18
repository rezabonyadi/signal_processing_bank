import signalscharacterisation.FeaturesImplementations as FeImpl


class SignalsFeatures:

    @staticmethod
    def call_features_by_indexes(features_indexes, x, settings):
        """
        :param features_indexes:
        :param x:
        :param settings:
        :return:
        """
        length = features_indexes.shape
        return_list = [dict() for i in range(length[0])]
        features_list = SignalsFeatures.get_features_list()
        for i in features_indexes:
            return_list[i] = SignalsFeatures.call_feature_by_name(features_list[i], x, settings)

        return return_list

    @staticmethod
    def get_features_list():
        return FeImpl.FeaturesImplementations.get_features_list()

    @staticmethod
    def call_feature_by_name(feature_name, x, settings):
        """
        :param feature_name:
        :param x:
        :param settings:
        :return:
        """
        feature = getattr(FeImpl.FeaturesImplementations, feature_name)
        return feature(x, settings)


