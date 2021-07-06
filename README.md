# AlphaNet: Improved Training of Supernet with Alpha-Divergence
This repository contains our PyTorch training code, evaluation code and pretrained models for AlphaNet.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/alphanet-improved-training-of-supernet-with/neural-architecture-search-on-imagenet)](https://paperswithcode.com/sota/neural-architecture-search-on-imagenet?p=alphanet-improved-training-of-supernet-with)

Our implementation is largely based on [AttentiveNAS](https://arxiv.org/pdf/2011.09011.pdf). 
To reproduce our results, please first download the [AttentiveNAS repo](https://github.com/facebookresearch/AttentiveNAS), and use our *train\_alphanet.py* for training and *test\_alphanet.py* for testing.

For more details, please see [AlphaNet: Improved Training of Supernet with Alpha-Divergence](https://arxiv.org/pdf/2102.07954.pdf) by Dilin Wang, Chengyue Gong, Meng Li, Qiang Liu, Vikas Chandra.

If you find this repo useful in your research, please consider citing our work and [AttentiveNAS](https://arxiv.org/pdf/2011.09011.pdf):

```BibTex
@article{wang2021alphanet,
  title={AlphaNet: Improved Training of Supernet with Alpha-Divergence},
  author={Wang, Dilin and Gong, Chengyue and Li, Meng and Liu, Qiang and Chandra, Vikas},
  journal={arXiv preprint arXiv:2102.07954},
  year={2021}
}

@article{wang2020attentivenas,
  title={AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling},
  author={Wang, Dilin and Li, Meng and Gong, Chengyue and Chandra, Vikas},
  journal={arXiv preprint arXiv:2011.09011},
  year={2020}
}
```

## Evaluation
To reproduce our results:
- Please first download our [pretrained AlphaNet models](https://drive.google.com/file/d/1CyZoPyiCoGJ0qv8bqi7s7TQRUum_8FeG/view?usp=sharing) from a Google Drive path and put the pretrained models under your local folder *./alphanet_data*

- To evaluate our pre-trained AlphaNet models, from AlphaNet-A0 to A6, on ImageNet with a single GPU, please run:

    ```python
    python test_alphanet.py --config-file ./configs/eval_alphanet_models.yml --model a[0-6]
    ```

    Expected results:
    
    | Name  | MFLOPs  | Top-1 (%) |
    | :------------ |:---------------:| -----:|
    | AlphaNet-A0      | 203 | 77.87 |
    | AlphaNet-A1     | 279 | 78.94 |
    | AlphaNet-A2     | 317 | 79.20 |
    | AlphaNet-A3    | 357 | 79.41 |
    | AlphaNet-A4     | 444 | 80.01 |
    | AlphaNet-A5 (small)     | 491 | 80.29 |
    | AlphaNet-A5 (base)    | 596 | 80.62 |
    | AlphaNet-A6     | 709 | 80.78 |
    
- Additionally, [here](https://drive.google.com/file/d/1NgZhJy8MJnuxjXkJ0gfnBGyrUVYwbAmx/view?usp=sharing) is our pretrained supernet with KL based inplace-KD and [here](https://drive.google.com/file/d/1rj1opDnlBD2_8ZV--LUSn8HXWfhiMdu8/view?usp=sharing) is our pretrained supernet without inplace-KD. 

## Training
To train our AlphaNet models from scratch, please run:
```python
python train_alphanet.py --config-file configs/train_alphanet_models.yml --machine-rank ${machine_rank} --num-machines ${num_machines} --dist-url ${dist_url}
```
We adopt SGD training on 64 GPUs. The mini-batch size is 32 per GPU; all training hyper-parameters are specified in [train_alphanet_models.yml](configs/train_alphanet_models.yml).

## Evolutionary search
In case you want to search the set of models of your own interest - we provide an example to show how to search the Pareto models for the best FLOPs vs. accuracy tradeoffs in _parallel_supernet_evo_search.py_; to run this example:
```python
python parallel_supernet_evo_search.py --config-file configs/parallel_supernet_evo_search.yml 
```

## License
AlphaNet is licensed under CC-BY-NC.

## Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more info.


