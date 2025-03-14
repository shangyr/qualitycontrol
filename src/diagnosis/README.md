# How to Run
## Environment
- `python 3.9`
- `pip install -r requirements.txt`
- `sh install_pyg.sh`

## Code
change PROJECT_DIR in `parsers.py`

## Run
### train
#### 单卡
`python main.py`
#### 多卡
`python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main.py`

### test
调用方法详见`test_predictor.py`
如果在其他地方调用，在调用前加入

```python
import sys
sys.path.append(PROJECT_DIR+'code')
```

