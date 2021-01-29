user_name=whoami
#temp_dir = 
#echo "build git"
#git_version = 
#echo "build cmake"
#cmake_version = 
#echo "build axel"
#axel_version = 
#echo "build wget"
#wget_version = 
echo "build personal shadowsocks"
remote_ip = 
save_dir = 
git clone https://github.com/jihuacao/UbuntuConfig.git
cd ./UbuntuConfig/shadowsocks/
while read line
do
    echo $line
    [[ "$line" =~ "kcptun client:" ]] && echo "asdasd"
    [[ "$line" =~ "kcptun client:" ]]
    echo --------$t
done < ./example_config.md
cd ../../
#echo "build vim"
#git clone https://github.com/jihuacao/UbuntuConfig.git
#cp ./UbuntuConfig/.vimrc /home/
#echo "build bashrc"
#echo "build miniconda"
#miniconda_version = 
echo "build miniconda env"
env_python_version =
conda create --n $env_python_version python=$env_python_version
#echo "build gpu driver"
##http://yysfire.github.io/linux/build-and-install-official-NVIDIA-driver-manually-on-ubuntu-12.04.html
#echo "build cuda"
#cuda_version =
##todo: export CUDA_HOME=$CUDA_HOME:/data2/cuda-backup/cuda-10.0
##todo: export PATH=$PATH:/data2/cuda-backup/cuda-10.0/bin
##todo: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data2/cuda-backup/cuda-10.0/lib64
#echo "build cudnn"
#cudnn_version =
echo "build gloo"
gloo_version=
git clone https://github.com/solo-io/gloo.git
cd gloo
git checkout $gloo_version
git submodule sync
git submodule update --init --recursive
echo "build nccl"
#https://github.com/NVIDIA/nccl/tree/master
#NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved 
#when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.
nccl_version = 
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout $nccl_version
git submodule sync
git submodule update --init --recursive
make -j src.build CUDA_HOME=$CUDA_HOME
sudo apt install build-essential devscripts debhelper fakeroot
make pkg.debian.build
#test the nccl
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j12 CUDA_HOME=/usr/local/cuda
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
echo "build mpi"
mpi_version=
git clone https://github.com/open-mpi/ompi.git
cd ompi
git submodule sync
git submodule update --init -- recursive
./configure --prefix=~/.local/ |& tee config.out
make -j 8 |& tee make.out
make install |& tee install.out
cd ../
#echo "build torch"
#torch_version=
##https://github.com/pytorch/pytorch
##https://github.com/pytorch/pytorch#from-source
##need 1.6 1.7(python >= 3.6.2) 1.5(python >= 3.5.6)
#git clone https://github.com/pytorch/pytorch.git
#cd pytorch
#git checkout $torch_version
#git submodule sync
#git submodule update --init -recursive
#pytorch_version=
#proxychains pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
#export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#python setup.py install
#cd ../
#echo "build torch"
#torch_version=
##https://github.com/pytorch/pytorch
##https://github.com/pytorch/pytorch#from-source
##need 1.6 1.7(python >= 3.6.2) 1.5(python >= 3.5.6)
#git clone https://github.com/pytorch/pytorch.git
#cd pytorch
#git checkout $torch_version
#git submodule sync
#git submodule update --init -recursive
#pytorch_version=
#proxychains pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
#export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#python setup.py install
#cd ../
#echo "build torchvision"
##https://github.com/pytorch/vision
##torch	            torchvision	    python
##master/nightly	master/nightly	>=3.6
##1.7.0	            0.8.0	        >=3.6
##1.6.0	            0.7.0	        >=3.6
##1.5.1	            0.6.1	        >=3.5
##1.5.0	            0.6.0	        >=3.5
##1.4.0	            0.5.0	        ==2.7, >=3.5, <=3.8
##1.3.1	            0.4.2	        ==2.7, >=3.5, <=3.7
##1.3.0	            0.4.1	        ==2.7, >=3.5, <=3.7
##1.2.0	            0.4.0	        ==2.7, >=3.5, <=3.7
##1.1.0	            0.3.0	        ==2.7, >=3.5, <=3.7
##<=1.0.1	        0.2.2	        ==2.7, >=3.5, <=3.7
#torchvision_version=$depend on the torch
#git clone https://github.com/pytorch/vision.git
#cd vision
#git checkout $torchvision_version
#pip install pillow
#python setup.py install
#cd ../
#echo "build torchaudio"
##torch	        torchaudio	    python
##master/nightly	master/nightly	>=3.6
##1.7.0	        0.7.0	        >=3.6
##1.6.0	        0.6.0	        >=3.6
##1.5.0	        0.5.0	        >=3.5
##1.4.0	        0.4.0	        ==2.7, >=3.5, <=3.8
#torchaudio_version=$depend_on_torch
#git clone https://github.com/pytorch/audio.git
#cd audio
#git checkout $torchaudio_version
#python setup.py install
#cd ../
#echo "build torchtext"
##PyTorch version	torchtext version	Supported Python version
##nightly build	    master	            3.6+
##1.7	            0.8	                3.6+
##1.6	            0.7	                3.6+
##1.5	            0.6	                3.5+
##1.4	            0.5	                2.7, 3.5+
##0.4 and below	    0.2.3	            2.7, 3.5+
#torchtext_version=$depend_on_torch
#git clone https://github.com/pytorch/text.git
#cd text
#git checkout $torchtext_version
#git submodule update --init --recursive
##linux
#python setup.py clean install
#cd ../
#echo "build tf"
#tf_version = 
#echo "build mxnet"
#mxnet_version = 
#echo "build horovod"
##https://github.com/horovod/horovod
### problem record: 2020/11/25 pytorch1.7 horovod0.19.5 使用python setup.py install出Internal Error reported in 
###     method “allreduce”， Requested ReadyEvent with GPU device but not compiled with CUDA 的错误，
###     其实是因为horovod0.19.5没有适应最新版的torch1.7，出现了setup.py中判断torch是否cuda版还是cpu版，我直接在is_torch_cuda_v2
###     中出现exception中直接return True达到直接判断为cuda版的情况，已经提交了issue给horovod，等待回复
### record 可以使用horovodrun --check-build查看编译情况
horovod_version= 
git clone https://github.com/horovod/horovod.git
git submodule sync
git submodule update --init --recursive
cd horovod
mkdir build
cd build
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_TENSORFLOW=1 \
HOROVOD_WITH_PYTORCH=1 proxychains pip install --no-cache-dir horovod==$horovod_version
make -j8
make install
#echo "build Putil"
#echo "build learning"
#echo "build note"
#echo "Toy"