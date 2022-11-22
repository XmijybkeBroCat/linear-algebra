def linear_regression(_X: list[int, float], _Y: list[int, float]):
    ave_x = sum(_X) / len(_X)
    ave_y = sum(_Y) / len(_Y)
    D_x = sum(((i - ave_x) ** 2) for i in _X)
    D_y = sum(((i - ave_y) ** 2) for i in _Y)
    dxdy = sum(((_X[i] - ave_x) * (_Y[i] - ave_y)) for i in range(len(_X)))
    k = dxdy / D_x
    b = ave_y - k * ave_x
    r = dxdy * (D_y * D_x) ** -0.5
    return k, b, r
