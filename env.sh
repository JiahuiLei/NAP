conda remove -n nap-gcc9 --all -y
conda create -n nap-gcc9 gcc_linux-64=9 gxx_linux-64=9 python=3.9 -y
source activate nap-gcc9

which python
which pip
CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
CPP=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
$CC --version
$CXX --version

# install pytorch
echo ====INSTALLING PyTorch======
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# # install pytorch3d
echo ====INSTALLING=PYTORCH3D======
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d=0.7.4 -c pytorch3d -y # 0.6.1

# # Install Pytorch Geometry
conda install pyg -c pyg -y

# install requirements
pip install cython
pip install -r requirements.txt
# pip install pyopengl==3.1.5
pip install numpy==1.23

# # build ONet Tools
python setup.py build_ext --inplace
python setup_c.py build_ext --inplace # for kdtree in cuda 11