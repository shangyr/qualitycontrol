

# 病历质检算法部分

待完成 `src/main_algorithm/*.py`中，所有类的check函数。

# 目前提供工具

1. 医疗实体提取，通过`self.match_model.find_entities(sentence)` 得到结果
2. 时间地点提取，通过`self.ner_model.find_entities(sentence)`得到结果
3. 句法分析，通过`self.spacy_model.find_entities(sentence)`得到结果
4. 关系分析，通过`self.graph_model.search_link_paths(entity1,entity2)`得到结果
5. 相关关系查找，通过`self.graph_model.search_paths(entity1)`得到结果
# 配置
python 3.9
```
pip install tqdm
pip install torch==1.11.0
pip install transformers==4.35.2
pip install jieba
pip install fuzzywuzzy
pip install spacy
pip install pandas
pip install wandb
pip install gensim==4.2.0
pip install scikit-opt==0.6.6
python -m spacy download zh_core_web_sm
```

到网站 https://data.pyg.org/whl/ 下载对应版本的下面四个文件，然后通过pip安装，或者参考 https://zhuanlan.zhihu.com/p/659091190?utm_id=0/ 进行安装
whl版本不一定要对应，但需要符合cuda版本、python版本和电脑版本（win/linux）
```
pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

pip install torch-geometric==2.2.0
```


在test/test_algorithm.py中设置项目地址 `PROJECT_DIR`, 比如我的项目在`C:\\Users\\Administrator\\Desktop\\qualitycontrol_service` ，这个目录下包含data、images、src等目录，PROJECT_DIR就需要设置为`C:\\Users\\Administrator\\Desktop\\qualitycontrol_service`。
同样在test/test_algorithm.py中设置预测模型地址地址`DIAGNOSIS_DIR`，如上述项目位置不变，`DIAGNOSIS_DIR`就需要设置为 `C:\\Users\\Administrator\\Desktop\\qualitycontrol_service\\src\\diagnosis`。

linux需安装git-lfs
```bash
git lfs install
```

# 算法测试

`python test/test_algorithm.py`

### 运行截图
![alt text](images/%E6%88%AA%E5%9B%BE1.png)

# 扩充图谱

### 已通过一些来源对原图谱进行扩充
形成v2版本实体文件和图谱文件

