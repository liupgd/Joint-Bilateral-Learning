# Joint Bilateral Learning
[Github link](https://github.com/mousecpn/Joint-Bilateral-Learning)  
This repository is an unofficial implementation in PyTorch for the paper:

[Joint Bilateral Learning for Real-time Universal Photorealistic Style Transfer](https://arxiv.org/abs/2004.10955)



### Dependencies

- Python 3.7.2
- PyTorch 1.2
- CUDA10.0 and cuDNN
- pytorch-lightning(newest, 1.04rc1)



### Train

```
$ python main.py --cont_img_path <path/to/cont_img> --style_img_path <path/to/style_img> --batch_size 8
```



### Test

```
$ python test.py --cont_img_path <path/to/single_cont_img> --style_img_path <path/to/single_style_img> --model_checkpoint <path/to/model checkpoint>
```

### Use pytorch-lightning
* Train  
```
python main_pl.py --gpus=2,3,4,5 --batch_size=6 --num_workers=6 --vgg_ckp_file=<vgg checkpoint file> --content_path=<content images path> --style_path=<style images path> --val_content=<validation content images path> --val_style=<validation style images path>
```

