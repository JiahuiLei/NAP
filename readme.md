# *NAP*: Neural Articulated Object Prior

[project page](https://www.cis.upenn.edu/~leijh/) Neurips-2023

**[2023.Oct.21] Note: because we are a little busy recently, we are still working on the full releaseing, currently the repo is a preview, supporting basic training and inference**

## Install

Run `bash env.sh`, this will create a conda environment named `nap-gcc9` and install all dependencies. This script is tested with Ubuntu 20.04 and cuda 11.7.

## Prepare data and checkpoints

Currently, we only release the pre-processed training data for articulated objects. The part shape prior data is not released yet.

- Download pre-processed data from [link](https://drive.google.com/file/d/1XCE0YadL8LaXCJCkl8TTnYAIlPjIZnsB/view?usp=sharing). Unzip it and form the directory structure as follows:

    ```
    PROJECTROOT/data
    ├── partnet_mobility_graph_mesh
    └── partnet_mobility_graph_v4
    ```

- Download the pretrained checkpoint (necessary for training NAP because it contains the shape prior network weights) from [link](https://drive.google.com/file/d/1v5PYwWCevhoRjjn8-Wh5ThVIOr6aouXQ/view?usp=sharing). Put them under `PROJECTROOT/log/`:
  
    ```
    PROJECTROOT/log
    ├── s1.5_partshape_ae
    └── v6.1_diffusion
    ```

- Optionally, you can download the evaluation output example from [link](https://drive.google.com/file/d/11sxvJ4sP3TOEHRLVw44-uO_u2dPpuURg/view?usp=sharing). Unzip it and put it under `PROJECTROOT/log/test/`:

    ```
    PROJECTROOT/log/test
    ├── G
    ├── ID_D_matrix
    ├── PCL
    └── Viz
    ```

Again, here are the downloading links

[training-data-download](https://drive.google.com/file/d/1XCE0YadL8LaXCJCkl8TTnYAIlPjIZnsB/view?usp=sharing)

[checkpoint-download](https://drive.google.com/file/d/1v5PYwWCevhoRjjn8-Wh5ThVIOr6aouXQ/view?usp=sharing)

[test-output-example-download](https://drive.google.com/file/d/11sxvJ4sP3TOEHRLVw44-uO_u2dPpuURg/view?usp=sharing)

## Training

One training example is:

```shell
python run.py --config ./configs/nap/v6.1_diffusion.yaml -f
```
You can also check the `.vscode/launch.json`.

## Testing

Computing the metrics takes some time, please see [`eval/readme_eval.md`](./eval/readme_eval.md) for details.


# Note

**If you find this repo useful, please cite our paper. Thank you!**

**As well as the original PartNet-Mobility dataset [their website](https://sapien.ucsd.edu/)**
