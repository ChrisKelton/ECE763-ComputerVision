# ECE763

## 🛠️ Installation

- Download and install [Anaconda](https://www.anaconda.com/download).

- Create a conda virtual environment with the name, e.g., `ece763`, and activate it:

  ```bash
  # list existing env
  # conda env list
  # remove an env
  # conda env remove -n env_name

  conda create -n ece763 python=3.11.7 -y
  conda activate ece763
  ```

  We will use ```python3.11 -m``` in running pip or other installations below.

- Put this code in a folder, e.g., `~/ece763`

    ```bash
    cd ~/ece763
    ```

- [**Linux and Windows w/ GPU**] Install PyTorch:

  For examples, to install `torch==2.1.2` with `CUDA==11.8`:

  ```bash
  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

  Or

  ```bash
  python3.11 -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- [**Linux and Windows w/ CPU Only**] Install PyTorch:

  ```bash
  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cpuonly -c pytorch
  ```

  Or

  ```bash
  python3.11 -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
  ```

- [**Mac OSX**] Install PyTorch:

  ```bash
  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch
  ```

  Or

  ```bash
  python3.11 -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
  ```

- Install other requirements:

  ```bash
  python3.11 -m pip install -r ./requirements.txt
  ```

- Install other packages

    MMEngine

    ```bash
    python3.11 -m pip install mmengine==0.10.3
    # test it
    python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
    ```

    [**GPU**] MMCV (for cu118 and torch2.1)

    ```bash
    python3.11 -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
    ```

    [**CPU**]  MMCV (for cu118 and torch2.1)

    ```bash
    python3.11 -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
    ```

    Create a 3rdparty folder under the code folder

    ```bash
    # cd to your work dir
    mkdir 3rdparty
    cd 3rdparty
    ```

  Torchvision v0.16.2

    ```bash
    # make sure you are in the 3rdparty folder 
    git clone https://github.com/pytorch/vision.git ./torchvision
    cd torchvision
    git checkout v0.16.2
    cd ..
    ```

    Pytorch-image-models (timm) 0.9.12

    ```bash
    # make sure you are in the 3rdparty folder
    git clone https://github.com/huggingface/pytorch-image-models.git
    cd pytorch-image-models
    git checkout v0.9.12
    cd ..
    ```

    > "--device cpu" below can be replaced by, e.g., "--device cuda:0"

    MMPretrain v1.2.0

    ```bash
    # make sure you are in the 3rdparty folder
    git clone https://github.com/open-mmlab/mmpretrain.git
    cd mmpretrain
    git checkout v1.2.0
    python3.11 -m pip install -v .
    cd ..
    ```

    ```bash
    # test it
    cd mmpretrain
    python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cpu
    cd ..
    ```

    MMDetection v3.3.0

    ```bash
    # make sure you are in the 3rdparty folder
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmpdetection
    git checkout v3.3.0
    python3.11 -m pip install -v .
    cd ..
    ```

    ```bash
    # test it (wget is needed)
    cd mmpdetection
    python3.11 -m pip install ftfy==6.1.3 regex
    mkdir checkpoints
    cd checkpoints
    wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
    cd ..
    python demo/image_demo.py demo/demo.jpg ./configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py --weights ./checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
    cd ..
    ```

    MMSegmentation v1.2.2

    ```bash
    # make sure you are in the 3rdparty folder
    git clone https://github.com/open-mmlab/mmsegmentation.git
    cd mmpsegmentation
    git checkout v1.2.2
    python3.11 -m pip install -v .
    cd ..
    ```

    ```bash
    # test it (wget is needed)
    cd mmsegmentation
    mkdir checkpoints
    cd checkpoints
    wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth
    cd ..
    python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py ./checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cpu --out-file result.jpg
    cd ..
    ```

    MMagic v1.2.0

    ```bash
    # make sure you are in the 3rdparty folder
    git clone https://github.com/open-mmlab/mmagic.git
    cd mmagic
    git checkout v1.2.0
    python3.11 -m pip install -v .
    cd ..
    ```

    ```bash
    cd mmagic
    # resolve an issue of diffusers==0.26.1 https://github.com/open-mmlab/mmagic/issues/2110
    python3.11 -m pip install diffusers==0.24.0
    # test it (wget is needed)
    # to speed up cpu inference
    python3.11 -m pip install accelerate==0.26.1
    python demo/mmagic_inference_demo.py \
    --model-name stable_diffusion \
    --device cpu \
    --text "A panda is having dinner at KFC" \
    --result-out-dir ./output/sd_res.png
    cd ..
    ```

    MMPose v1.3.1

    ```bash
    # make sure you are in the 3rdparty folder
    git clone https://github.com/open-mmlab/mmpose.git
    cd mmpose
    git checkout v1.3.1
    python3.11 -m pip install -v .
    cd ..
    ```

    ```bash
    cd mmpose
    python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py \
    https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --show --draw-heatmap --det-cat-id=15 \
    --device cpu
    ```
