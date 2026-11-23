# Take a Step Back: A Dual-Granularity Raw Feature Empowered Intermediate Fusion Graph Convolutional Network for Multimodal Recommendation

<!-- PROJECT LOGO -->

## Introduction

This is the Pytorch implementation for our DIGCN paper:

>Take a Step Back: A Dual-Granularity Raw Feature Empowered Intermediate Fusion Graph Convolutional Network for Multimodal Recommendation

## Environment Requirement
- Python 3.9
- Pytorch 2.1.0

## Dataset

We provide four processed datasets: Baby, Sports, Pet, Office. [Link](https://drive.google.com/drive/folders/18SAyycFhoWxfNvrl1drMmQvlodQNsF6J)

## Training
  ```
  cd ./src
  python main.py
  ```

## Citing DIGCN

If you find DIGCN useful in your research, please consider citing our [paper](https://arxiv.org/pdf/2611.19974v1).

```
@article{xu2026take,
  title={Take a Step Back: A Dual-Granularity Raw Feature Empowered Intermediate Fusion Graph Convolutional Network for Multimodal Recommendation},
  author={Xu, Jinfeng and Chen, Zheyu and Yang, Shuo and Li, Jinze and Yuan, Wenhao and Peng, Ziyue and Wu, Puzhen and Li, Shujie and Chen, Jian and Lin, Zheng and others},
  journal={arXiv preprint arXiv:2611.19974},
  year={2026}
}
```


## Acknowledgement

The structure of this code is based on [MRLib](https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library) and [MMRec](https://github.com/enoche/MMRec). Thank for their work.

*We will contribute this work to [MRLib](https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library), so that it can be used directly for fair benchmarking.*