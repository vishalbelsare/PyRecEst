def f(n):
    value = n
    if isinstance(value, str):
        raise ValueError('bad')
    return int(value)
