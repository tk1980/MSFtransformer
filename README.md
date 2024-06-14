# Mean-Shift Feature Transformer

The Pytorch implementation for the CVPR2024 paper of "[Mean-Shift Feature Transformer](https://staff.aist.go.jp/takumi.kobayashi/publication/2024/CVPR2024.pdf)" by [Takumi Kobayashi](https://staff.aist.go.jp/takumi.kobayashi/).

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2024cvpr,
  title={Mean-Shift Feature Transformer},
  author={Takumi Kobayashi},
  booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Introduction

Transformer [1] is a key building block for recent advanced neural networks.
It is widely applied to various fields including CV, NLP and SP. 
In the transformer, token features are effectively transformed by means of (multi-head) attention.
So, it is vital to shed light on the feature transformation mechanism of transformer, for further development.
While many works mainly focus on attention, we analyze the mechanism of whole transformer process from a viewpoint of feature transformation based on token distribution.
The analysis clarifies analogy between transformer and mean-shift [2], a classical clustering approach in PR, which inspires us to propose mean-shift feature (MSF) transformer.
We also present an efficient grouped projection to reduce parameter sizes of the projection matrices in the transformers while retaining performance.
For more detail, please refer to our [paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2024/CVPR2024.pdf).

<img width=400 src="https://github.com/tk1980/MSFtransformer/assets/53114307/c45e1f57-bea6-422f-810e-41d94fa09bbd">

## Implementation

These codes are based on [vit_pytorch](https://github.com/lucidrains/vit-pytorch/vit_pytorch/simple_vit.py) for building SimpleViT models [3] and [pytorch classification codes](https://github.com/pytorch/vision/tree/main/references/classification) for training the models.

## Usage

Core of the MSF transformer is implemented by `AttentionMSF` class in `vit_pytorch/attention_module.py`.

### Training
For example, SimpleViT-S with MSF is trained on such as ImageNet dataset by means of distributed training over 4 GPUs over 100 epochs with 1024(=256*4) batch size;
```bash
torchrun --nproc_per_node=4 train.py --data-path <PATH_TO_IMAGENET> --output-dir <PATH_TO_OUTPUT> \
    --model SimpleViTS_msf --epochs 100 --batch-size 256 \
    --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear --lr-warmup-epochs 10 --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 0.2 --auto-augment three --ra-sampler
```
The MSF with 2-group projection is also trained by
```bash
torchrun --nproc_per_node=4 train.py --data-path <PATH_TO_IMAGENET> --output-dir <PATH_TO_OUTPUT> \
    --model SimpleViTS_msf-g2 --epochs 100 --batch-size 256 \
    --opt adamw --lr 0.001 --wd 0.05 --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear --lr-warmup-epochs 10 --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 0.2 --auto-augment three --ra-sampler
```

Note that ImageNet dataset must be downloaded at `<PATH_TO_IMAGENET>` before training.


## Results

#### ImageNet

| Method  | Param (M) | Acc (%) |
|---|---|---|
| Transformer | 21.91 | 78.98  | 
| MSF-Transformer | 23.68| 79.79 |
| MSF-Transformer w/ 2-group proj. | 20.14| 79.55 |

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017.

[2] Dorin Comaniciu and Peter Meer. Mean shift: A robust approach toward feature space analysis. IEEE TPAMI, 24(5): 603–619, 2002.

[3] Lucas Beyer, Xiaohua Zhai, and Alexander Kolesnikov. Better plain vit baselines for ImageNet-1k. arXiv:2205.01580, 2022.

## Contact
takumi.kobayashi (At) aist.go.jp