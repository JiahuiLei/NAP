# Standalone Evaluation Pipeline

Now this sub-directory is not fully standalone, which has dependencies under the project root, should clean the dependency there later.

1. Inference trained model to generate objects: `gen_diffusion.ipynb`. This notebook assume you have a trained model saved under log dir. After running this notebook, you will see two folders under `log/test/`: `G` saved all generated objects. `Viz` have some gifs visualization. **You should also have to run `save_gt.ipynb`** to save the ground truth objects for evaluation.

2. Sample point clouds from generated object with `sample_pcl.py`. Here `sample_pcl.sh` is an example. After running this, you will see sampled point cloud npz files under `PCL` dir

3. Compute the instantiation distance (very slow) based on the saved pcl files with `instantiation_distance.py` and example in `compute_id.sh`. After several hours GPU computing, you will see the distance matrices saved under`ID_D_matrix`.

4. Finally, you can compute the metrics from the saved distance matrices with `compute_metrics.ipynb`.