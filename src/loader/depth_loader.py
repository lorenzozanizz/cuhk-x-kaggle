from .loading_utils import load_vision


def load_depth(d, t):
    """

    :param d:
    :param t:
    :return:
    """
    return load_vision(d, t, 3)