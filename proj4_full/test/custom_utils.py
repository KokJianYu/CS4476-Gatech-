from functools import reduce
import torch
import numpy as np
import os

use_cuda = True and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tensor_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.set_default_tensor_type(tensor_type)

torch.backends.cudnn.deterministic = True
torch.manual_seed(333) #do not change this, this is to ensure your result is reproduciable

def loadbin(filename, device=device):
    with open(filename + '.dim') as file:
        dim = file.readlines()
        dim = np.array([int(x.strip()) for x in dim])
    #print(dim)
    size_1d = reduce(lambda x, y: x*y, dim)
    #print(dim)
    if os.path.exists(filename + '.type'):
        with open(filename + '.type') as file:
            type_ = file.readlines()
            type_ = [x.strip() for x in type_]
            assert len(type_) == 1
            type_ = type_[0]
    else:
        type_ = 'float32'

    if type_ == 'float32':
        x = torch.FloatTensor(torch.FloatStorage.from_file(filename,size=size_1d)).to(device)
    elif type_ == 'int32':
        x = torch.IntTensor(torch.IntStorage.from_file(filename,size=size_1d)).to(device)
    elif type_ == 'int64':
        x = torch.LongTensor(torch.LongStorage.from_file(filename,size=size_1d)).to(device)
    else:
        raise ValueError
    return x.reshape(tuple(dim))

def DataLoader(data_dir):
  X = []
  dispnoc = []
  height = 1500
  width = 1000

  for n in range(1,10):
      XX = []
      light = 1
      while True:
          fname = f'{data_dir}/x_{n}_{light}.bin'

          if not os.path.exists(fname):
              break
          loaded_data = loadbin(fname)
          #print(fname)
          if len(loaded_data) !=0:

              if len((loaded_data).size()) == 4:
                  loaded_data = loaded_data.unsqueeze(0)

              XX.append(loaded_data)
          light = light + 1

      #break
      X.append(XX)
      fname = f'{data_dir}/dispnoc{n}.bin'
      if os.path.exists(fname):
          dispnoc.append(loadbin(fname))
      else:
          dispnoc.append([])
  return X, dispnoc

def verify(function, argument) -> str:
  """ Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
    - function: Python function object
    Returns:
    - string
  """
  try:
    function(argument)
    return "\x1b[32m\"Correct\"\x1b[0m"
  except (AssertionError,RuntimeError) as e:
    print(e)
    return "\x1b[31m\"Wrong\"\x1b[0m"

def test_mcnet(mcnet):
  assert mcnet(torch.Tensor(2,1,11,11)).shape == torch.Size([1, 1])
  temp_net = load_model(mcnet.net,'mc_cnn_network_pretrain_ws11.pth', strict=True)


def test_extendednet(extendednet):
  mcnet_params = 123456
  extendednet_params = sum(p.numel() for p in extendednet.parameters())
  assert extendednet_params > mcnet_params


def save_model(network, path):
    torch.save(network.state_dict(),path)
def load_model(network, path, device='cpu', strict=True):
    network.load_state_dict(torch.load(path,map_location=torch.device(device)),strict=strict)
    return network

def test_gen_patch(gen_patch):
  image = torch.tensor([[[156, 219,  57, 188, 105],
        [198, 148,  54,  74, 236],
        [196,   3, 147,  68,  81],
        [ 85, 132, 227,  27, 225],
        [250,  28, 168, 118,  12]],

       [[ 83,  31, 221, 247, 236],
        [102, 229,   9, 221, 179],
        [ 49, 241, 223, 234,  48],
        [ 40, 167, 173,  98, 246],
        [ 93, 165, 229, 144,  96]]])
  out = gen_patch(image,2,2,3)

  gt = torch.tensor([[[147.,  68.,  81.],
         [227.,  27., 225.],
         [168., 118.,  12.]],

        [[223., 234.,  48.],
         [173.,  98., 246.],
         [229., 144.,  96.]]])

  ##test corner case
  out = gen_patch(image,4,4,3)

  gt = torch.tensor([[[12.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]],

        [[96.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]])
  assert torch.all(torch.isclose(out,gt))

def get_disparity(nnz,ind):
  img = nnz[ind,0]
  dim3 = nnz[ind,1]
  dim4 = nnz[ind,2]
  d = nnz[ind,3]
  return img, dim3, dim4, d