import os
from feature_extract import extract

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


# check whether the dataset has been downloaded or not
def check_feature():
    npylist = ['./mfcc.npy', './label.npy', './temp.npy', './melsp.npy', './melframe.npy']
    print(npylist)
    print('hello')
    for npy in npylist:
        if os.path.exists(npy):
            continue
        else:
            extract()
    return True