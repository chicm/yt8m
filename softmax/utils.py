import os.path as osp
import pandas as pd
import settings

def get_classes_1001():
    df = pd.read_csv(osp.join(settings.META_DIR, 'classes_1001.csv'))
    c = df['class'].values.tolist()
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi

def get_classes_1000():
    df = pd.read_csv(osp.join(settings.META_DIR, 'classes_1000.csv'))
    c = df['class'].values.tolist()
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi

def accuracy(output, label, topk=(1,10)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res

def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias

if __name__ == '__main__':
    c, _ = get_classes()
    print(c[:5])
    print(len(c))
