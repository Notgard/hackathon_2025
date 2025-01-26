# Teratec Hackathon 2025

This repository contains the work of team FC ZiGaAnDa for the Teratec Hackathon.  
In this repo, you will find :
* The Monte-Carlo Black-Scholes optimized codes
* The Code Aster installation instructions
* PDFs containing the guidelines of the event
* As well as a folder containing various useful data files

## xSimd Installation Guide & compilation

This guide provides steps to install `xSimd` in a Conda environment. Also to compile our differents files.

### Installation Steps

1. **Install CMake**
   Install `cmake` in your Conda environment using the following command:

   ```bash
   conda install anaconda::cmake
   ```

2. **Clone the xSimd Repository**
   Clone the `xSimd` GitHub repository by running:

   ```bash
   git clone https://github.com/xtensor-stack/xsimd.git
   ```

3. **Navigate to the xSimd Directory**
   Change directory to the `xsimd` repository:

   ```bash
   cd xsimd
   ```

4. **Run CMake**
   Configure the build system by running the following command. Replace `your_install_prefix` with the desired installation path:

   ```bash
   cmake -D CMAKE_INSTALL_PREFIX=your_install_prefix .
   ```

5. **Install xSimd**
   Build and install `xSimd` using:

   ```bash
   make install
   ```

## SIMD Everywhere Installation (for AVX port on ARM architectures)
1. **Clone the xSimd Repository**
   Clone the `xSimd` GitHub repository by running:

   ```bash
   git clone https://github.com/xtensor-stack/xsimd.git
   ```

2. **Make sure the include in the code points to the headers in the cloned repository**
```c++
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "cloned_repository/simde/x86/avx2.h"
#include "cloned_repository/simde/x86/fma.h"
```

## Compilation lines

1. **Compilation line for initial version**
```bash
   g++ -O BSM_INIT.cxx -o BSM_INIT
   ```

2. **Compilation line for OpenMPI version**
```bash
   mpicxx -std=c++20 -march=armv8-a -mcpu=native -DREAL="float" -Wall -O3 -flto -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fopenmp -o BSM_MPI BSM_MPI.cxx -I$HOME/xsimd/include
   ```
3. **Compilation line for OpenMP version**
```bash
   g++ -std=c++20 -march=armv8-a -mcpu=native -DREAL="float" -Wall -O3 -flto -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fopenmp -o BSM_OPEN BSM_OPEN.cxx -I$HOME/xsimd/include
   ```

4. **Compilation line for randoms generated at compilation version**
```bash
   mpicxx -std=c++20 -march=armv8-a -mcpu=native -DSEED=[your_seed] -DREAL="float" -Wall -O3 -flto -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fopenmp -o BSM_RANDOM_COMPILATED BSM_RANDOM_COMPILATED.cxx -I$HOME/xsimd/include
   ```

5. **Compilation line for the ARM version with AVX 128 instrinsics**
Note: First make sure to load the armpl module (source the setup_env.sh script)
```bash
   g++ -std=c++20 -march=armv8-a -mcpu=native -DREAL="float" -I/opt/arm/armpl/include -larmpl_lp64 -larmpl -Wall -O3 -flto -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fopenmp -o BSM_AVX_ARM_128 BSM_AVX_ARM_128.cxx
   ```

6. **Compilation line for the ARM version with AVX 256 instrinsics**
Note: First make sure to load the armpl module (source the setup_env.sh script)
```bash
   g++ -std=c++20 -march=armv8-a -mcpu=native -DREAL="float" -I/opt/arm/armpl/include -larmpl_lp64 -larmpl -Wall -O3 -flto -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fopenmp -o BSM_AVX_ARM_256 BSM_AVX_ARM_256.cxx
   ```

7. **Compilation line for the ARM version with XSIMD**
Note: First make sure to load the armpl module (source the setup_env.sh script)
```bash
   g++ -std=c++20 -march=armv8-a -mcpu=native -DREAL="float" -I/opt/arm/armpl/include -larmpl_lp64 -larmpl -Wall -O3 -flto -ffast-math -funroll-loops -fomit-frame-pointer -ftree-vectorize -fopenmp -o BSM_XSIMD_ARM BSM_XSIMD_ARM.cxx -I$HOME/xsimd/include -L/tools/acfl/24.10/gcc-14.2.0_AmazonLinux-2/lib64 -Wl,-rpath,/tools/acfl/24.10/gcc-14.2.0_AmazonLinux-2/lib64
   ```
   
### Notes for compilation

- Ensure that your path to Xsimd is correct
- Choose your seed for the 4th compilation

### Notes for xSimd

- Ensure that your Conda environment is active before performing these steps.
- Replace `your_install_prefix` with the absolute path where you want `xSimd` to be installed.

For more information, refer to the [official xSimd GitHub repository](https://github.com/xtensor-stack/xsimd).

## Code Aster build with Docker

```bash
docker build -f Dockerfile.debian11 -t codeaster .
docker run codeaster
```