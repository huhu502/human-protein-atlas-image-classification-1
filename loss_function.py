from torch import sum, where, zeros

def f1_loss(input, target, epsilon=1E-8):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    tp = sum(input * target, dim=0) # size = [1, ncol]
    tn = sum((1 - target) * (1 - input), dim=0) # size = [1, ncol]
    fp = sum((1 - target) * input, dim=0) # size = [1, ncol]
    fn = sum(target * (1 - input), dim=0) # size = [1, ncol]
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    f1 = 2 * p * r / (p + r + epsilon)
    # f1 = where(f1 != f1, zeros(f1.size()), f1)
    return 1 - f1.mean()
