from scipy.stats import spearmanr


def spearman(x, y):
    assert len(x) == len(y)

    return spearmanr(x, y)