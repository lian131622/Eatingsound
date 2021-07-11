
# get the type and shape(if has one) of an variable
def var_info(var, show=False):
    print('-' * 50)
    print('type:', type(var))
    try:
        var.shape
    except AttributeError:
        print('no shape attribute')
    else:
        print('shape:', var.shape)
    try:
        len(var)
    except TypeError:
        print("no length")
    else:
        print('len:', len(var))
    if show:
        print(var)