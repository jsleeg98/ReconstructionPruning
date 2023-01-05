import sys
import torch

def device(dev_name):
    if dev_name == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(dev_name if torch.cuda.is_available() else 'cpu')
    return device

def list_check(mean, std, w_h):
    if len(mean) != 3:
        print("the length of mean must be 3")
        sys.exit()
    elif len(std) !=3:
        print("the length of std must be 3")
        sys.exit()
    elif len(w_h) !=2:
        print("the length of w, h must be 2")
        sys.exit()

    return mean, std, w_h[0], w_h[1]