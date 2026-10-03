"""Observed-cell Gaussian geometry supervision."""

def geometry_nll(aux, teacher, valid):
    error = (aux['mean'].float() - teacher.float()).square()
    nll = .5 * (error * (-aux['logvar'].float()).exp() + aux['logvar'].float())
    weight = valid[..., None].float()
    return (nll * weight).sum() / (3 * weight.sum().clamp_min(1))
