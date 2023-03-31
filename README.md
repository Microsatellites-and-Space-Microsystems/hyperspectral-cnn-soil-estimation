# hyperspectral-cnn-soil-estimation

This repository contains the code for reproducing the algorithms that we developed as part of the ESA's sponsored <a href="https://platform.ai4eo.eu/seeing-beyond-the-visible">#HYPERVIEW Competition</a>, challenging researchers to build AI-based methods for estimating soil parameters from hyperspectral images.  

The regression model consists of a CNN built upon a modified EfficientNet-lite0 [1-2]. With just 734 million parameters, the network is capable of processing up to about 180 frames per second on a Coral Dev Board Mini microcomputer.  

Our solution ranked 4th on the final leaderboard presented at the 2022 IEEE International Conference on Image Processing.

<div align="center">
    <img src="https://github.com/Microsatellites-and-Space-Microsystems/hyperspectral-cnn-soil-estimation/blob/main/images/efficientnet.png" alt="Illustration of the NN regression model obtained with VisualKeras [3]" width="700" height="auto" style="max-width:100%;">
    <br><em>Illustration of the NN regression model obtained with VisualKeras [3]</em>
</div>
<br>

The following data preprocessing steps have been implemented: normalization, tiling, flipping, and corruption with gaussian noise.

<div align="center">
    <img src="https://github.com/Microsatellites-and-Space-Microsystems/hyperspectral-cnn-soil-estimation/blob/main/images/preprocessing.png" alt="Sample band from a random         image: original (left), tiling and flipping (centre), noise corruption (right)" width="500" height="auto" style="max-width:100%;">
    <br><em>Sample band from a random image: original (left), tiling and flipping (centre), noise corruption (right)  </em>
</div>
<br>

<b>Authors</b> - Achille Ballabeni, Alessandro Lotti, Alfredo Locarini, Dario Modenini, Paolo Tortora.  
<a href="https://site.unibo.it/almasat-lab/en">u3S Laboratory @ Alma Mater Studiorum Università di Bologna</a>.

## How to use
The code has been tested on a Conda environment in WSL2 with Ubuntu 20.04, running on a Windows 11 computer with an RTX 3090 GPU. Nonetheless, the model is designed to be lightweight, and can be trained even without a GPU.

### 1) Install Miniconda  
#### - Linux / WSL2:  
`curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh`  
`bash Miniconda3-latest-Linux-x86_64.sh`  

To install WSL2 in Windows launch powershell as administrator, run `wsl --install`, and follow the instructions.  
Further information can be found at <a href="https://learn.microsoft.com/it-it/windows/wsl/install">this link</a>.  

#### - Windows native:  
- Download and install <a href="https://learn.microsoft.com/it-IT/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022">Visual Studio 2015, 2017, 2019 and 2022</a>  
- Download and install <a href="https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe">Miniconda</a>

#### - MacOS:
`curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh`  
`bash Miniconda3-latest-MacOSX-x86_64.sh`  

Note that currently there is no official GPU support for TensorFlow on MacOS.  

When setup is completed, it is suggested to reboot the system.

### 2) Setup the environment  
- Clone this repository:  
`git clone https://github.com/Microsatellites-and-Space-Microsystems/hyperspectral-cnn-soil-estimation`  
`cd hyperspectral-cnn-soil-estimation`  

- <b> If a dedicated NVIDIA GPU is available: </b>  
    - Create a new Conda environment:  
    `conda env create -f environment_GPU.yml`  
    
    - Reboot the system  
    
    - Activate the Conda environment:  
    `conda activate hyperspectral-cnn-soil-estimation`  
    
    - <b> Linux and WSL2 systems: </b>       
        - Execute the following commands to enable the GPU in TensorFlow:  
        `mkdir -p $CONDA_PREFIX/etc/conda/activate.d`  
        `echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`  
        
        - Reboot the system, and activate the Conda environment (`conda activate hyperspectral-cnn-soil-estimation`)
        
        - Check that the GPU is configured properly by executing:  
        `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`  

    - <b> Windows native: </b>  
        - Check that the GPU is configured properly by executing:  
        `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
    
- <b> If no GPU is available: </b>  
    - Create a new Conda environment:  
    `conda env create -f environment_CPU.yml`  
    
    - Reboot the system  
    
    - Activate the Conda environment:  
    `conda activate hyperspectral-cnn-soil-estimation`  

- Launch Jupyter running:  
`jupyter notebook`    

- Copy and paste the url `http://localhost:8888/tree` in a web browser.  

- Navigate to the desired notebook and enjoy the code.  

## Troubleshooting

- <b> On WSL2 / Linux </b>  

    - In case a permission denied error pops up, run:  
    `sudo chown -R your_linux_username /home/your_linux_username/hyperspectral-cnn-soil-estimation`  

## References

[1] Renjie Liu, <a href="https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html">"Higher accuracy on vision models with EfficientNet-Lite"</a>, TensorFlow Blog.

[2] Sebastian Szymanski, <a href="https://github.com/sebastian-sz/efficientnet-lite-keras">efficientnet-lite-keras</a>.

[3] Paul Gavrikov, <a href="https://github.com/paulgavrikov/visualkeras">Visualkeras</a>
