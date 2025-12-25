<!--
 * @Author: Long Chen
 * @Date: 2022-09-14 20:22:53
 * @LastEditors: Long Chen
 * @Description:
-->
1. install

```
cd emd_module
python setup.py install
 ```

2. use

```
from emd_module.emd_module import emdModule

def get_emd_loss(self, pred, gt, eps=1.0, iters=512):
    """
    pred and gt is B N 3
    """
    dis, _ = self.emd(pred, gt, eps, iters)
    dis = torch.mean(torch.sqrt(dis), dim=1)
    return torch.mean(dis)
```
