# Learning region representations with POIs and hierarchical graph infomax

> 本项目依附于发表在 `ISPRS Journal of Photogrammetry and Remote Sensing` 的论文： [Learning urban region representations with POIs and hierarchical graph infomax](https://doi.org/10.1016/j.isprsjprs.2022.11.021)
>
> 
>
> 在本研究中，作者提出了基于层次图的信息最大化方法 hierarchical graph infomax (HGI) 用于学习城市的区域表示，也可以说是向量嵌入（urban region representations/vector embeddings） 。整个方法只使用兴趣点（POI）数据，并且进行了完全无监督学习得到区域表示，这些表示可以用于各种各样的下游任务中。
>
> 总体的结构如下图所示：
>
> ![HGI](Figures/HGI.png)

## Quick start
通过本仓库，你可以：
- 学习如何利用预构建和采样好的深圳数据集进行区域表示嵌入;
- 将学习到的区域嵌入用于下游任务.

## Environment
- Python 3.8.13
- PyTorch 1.12.1
- Pytorch_geometric 2.1.0
- Scikit-learn 1.1.2
- Tdqm 4.61.4
- Pytorch_warmup 0.1.0

## Structure
本仓库有两个主要脚本： `train.py` 和 `evaluation.py`.


## `train.py`
`train.py` 用于学习 `region representations`. 通过 `--city` 参数指定数据集，比如：

````shell
python train.py --city shenzhen
````

学习到的 `region representations` 保存在 `./Emb` 文件夹中. 



`train.py` 可以指定的参数有：

- `--city`: 城市名称, e.g., "shenzhen". 
  - 我们提供了一个预构建和采样好的深圳数据集，可以下载该 [data](https://figshare.com/articles/dataset/Sub-sampled_dataset_for_Shenzhen_HGI_region_embedding_example_dataset_/21836496) `shenzhen_data.pkl` 并将其放置在 `Data` 文件夹内. 于是通过运行 `python train.py --city shenzhen` 即可开始训练.
- `--dim`:  `region representations` 的维数.
- `--alpha`: 用于平衡互信息（mutual information）的超参数 $\alpha$. 
- `--attention_head`: 指定聚合函数（aggregation function）中的多头注意力机制（attention heads）的头数.
- `--lr`: 学习率.
- `--max_norm`: the maximum norm of the gradient.
- `--gamma`: gamma in learning rate scheduler.
- `--warmup_period`: the warmup period, i.e., how many epochs for linear learning rate warmup.
- `--epoch`: the number of epochs.
- `--device`: the device to use, which can be `cpu` or `cuda` or `cuda:{}`.
- `--save_name`: 指定`region representations` 保存为什么名字.



如果你想构建自己的数据集来学习区域表示，那么你需要将构建好的数据对象存放在 `./Data` 文件夹，并且你的数据对象需要包含以下属性：
- `x`: POI initial features, which can be POI category embeddings (can even be one-hot vectors) or POI embeddings learned by other methods.
- `edge_index`: the POI graph structure, which is a tensor of shape $(2, E)$, where $E$ is the number of edges.
- `edge_weight`: the edge weights.
- `region_id`: the region id of each POI.
- `region_area`: the area proportion of each region in its city.
- `coarse_region_similarity`: the coarse similarity of each region with all other regions. This is a (N, N) matrix, where N is the number of regions.
- `region_adjacency`: the adjacency matrix of regions. 




## `evaluation.py`
`evaluation.py` 用于评估所学习到的区域表示 `region representations`. 现已支持两个地区:

- `xiamen`: 中国厦门 | Xiamen Island, China
- `shenzhen`: 中国深圳 | Shenzhen, China

支持三种下游任务：

- `uf`: urban function inference (this repo contains mocked ground truth data, the real ground truth data can be requested from [here](http://geoscape.pku.edu.cn/en.html))
- `pd`: population density estimation
- `hp`: housing price estimation



例如，你可以通过运行

````shell
python evaluation.py --city xiamen --task pd
````

来评估厦门的人口密度情况. 结果将打印在控制台(console).
![eval](Figures/eval.png)

## Embeddings
The learned region representations for Xiamen Island and Shenzhen are available in the `./Emb` folder. You can load them by `torch.load()`.

## Notes
部分从`POIs`层级到`Regins`层级的池化代码来自： [Set Transformer's GitHub repository](https://github.com/juho-lee/set_transformer).

## Citation

If you use the code in this project, please cite the paper the ISPRS Journal.
```
@article{huang2023hgi,
  title={Learning urban region representations with POIs and hierarchical graph infomax},
  author={Huang, Weiming and Zhang, Daokun and Mai, Gengchen and Guo, Xu and Cui, Lizhen},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={196},
  pages={134--145},
  year={2023}
}
```

## 联系原作者
Weiming Huang (南洋理工大学/Nanyang Technological University)

Email: weiming.huang@ntu.edu.sg
