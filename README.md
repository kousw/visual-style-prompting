## About

This repository is an experimental implementation of [Visual Style Prompting](https://arxiv.org/abs/2402.12974). **This is an unofficial implementation.**

## Note

This method seems to be able to extract and reflect the style of an image by swapping the key and value of the self-attention with the key and value of the reference image after the 24th layer of UpBlock in Unet.
From my experiments, it seems to be able to reflect some styles, but it may reflect excessive color schemes or broken details.
It is possible that I may have skipped over something in the paper's implementation, so this is just an experimental implementation.

Reference

<img src="./sample/ref/gogh.png" width="200"/>

Generated

`a cat sitting in a city`

<img src="./sample/00/out_sdxl.png" width="200"/>


## Environment

*Python* 3.10.9
*CUDA* 12.2

## Installation

```
python -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
```

## Inference (with SDXL)

**Command**

```
python inference_sdxl.py --guidance_scale 7.0 --num_inference_steps 50 --reference_image sample/ref2.png --prompt "low-poly stile cat, low-poly game art, polygon mesh, jagged blocky, wireframe edges, cnetered composition, simple"  --resolution 768 --num_samples 5
```

### Other Samples 

Reference

<img src="./sample/ref/ref2.png" width="200"/>

Generated(cat)

<img src="./sample/04/out_sdxl_0.png" width="200"/> <img src="./sample/04/out_sdxl_1.png" width="200"/> <img src="./sample/04/out_sdxl_2.png" width="200"/> <img src="./sample/04/out_sdxl_3.png" width="200"/> <img src="./sample/04/out_sdxl_4.png" width="200"/>


Generated(motorcycle)

<img src="./sample/03/out_sdxl_0.png" width="200"/> <img src="./sample/03/out_sdxl_1.png" width="200"/> <img src="./sample/03/out_sdxl_2.png" width="200"/> <img src="./sample/03/out_sdxl_3.png" width="200"/> <img src="./sample/03/out_sdxl_4.png" width="200"/>

Reference

<img src="./sample/ref/ref1.png" width="200"/>

Generated(cat)

<img src="./sample/01/out_sdxl_0.png" width="200"/> <img src="./sample/01/out_sdxl_1.png" width="200"/> <img src="./sample/01/out_sdxl_2.png" width="200"/> <img src="./sample/01/out_sdxl_3.png" width="200"/> <img src="./sample/01/out_sdxl_4.png" width="200"/>


Generated(motorcycle)

<img src="./sample/02/out_sdxl_0.png" width="200"/> <img src="./sample/02/out_sdxl_1.png" width="200"/> <img src="./sample/02/out_sdxl_2.png" width="200"/> <img src="./sample/02/out_sdxl_3.png" width="200"/> <img src="./sample/02/out_sdxl_4.png" width="200"/>
