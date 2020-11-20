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
#echo "build miniconda env"
#env_python_version =
#echo "build gpu driver"
##http://yysfire.github.io/linux/build-and-install-official-NVIDIA-driver-manually-on-ubuntu-12.04.html
#echo "build cuda"
#cuda_version =
##todo: export CUDA_HOME=$CUDA_HOME:/data2/cuda-backup/cuda-10.0
##todo: export PATH=$PATH:/data2/cuda-backup/cuda-10.0/bin
##todo: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data2/cuda-backup/cuda-10.0/lib64
#echo "build cudnn"
#cudnn_version =
#echo "build nccl"
##https://github.com/NVIDIA/nccl/tree/master
##NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved 
##when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.
#nccl_version = 
#echo "build horovod"
##https://github.com/horovod/horovod
#horovod_version= 
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
#echo "build Putil"
#echo "build learning"
#echo "build note"
#echo "Toy"