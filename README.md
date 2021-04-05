![alt text](https://github.com/agaz1985/ado/blob/main/imgs/logo.png?raw=true)

**ado** is a C++ Machine Learning Library with Python bindings, built from scratch by using the awesome [xtensor-stack](https://github.com/xtensor-stack).

---
***Are you in the right place ?***

If you are looking for an easy-to-read C++ implementation of the main machine learning algorithms, with a simple Python interface for experimenting with the code, you are in the right place.

If you are looking for an optimized, multi-platform, memory-efficient, heavely tested C++ machine learning library with a great community, I would recommend [TensorFlow C++ API](https://www.tensorflow.org/api_docs/cc) or [PyTorch C++ API](https://pytorch.org/cppdocs/).

# Getting Started
## Build & Install
1. Download and install Anaconda from https://www.anaconda.com/ 
2. Clone the ado repository:
   ```bash
    git clone git@github.com:agaz1985/ado.git
    ```
3. Prepare the building envoronment:
   ```bash
    cd ado
    ./install_requirements.sh
    cd ..
    mkdir build && cd build
    ```
4. Build the library, the bindings, the tests, the examples and the documentation:
   ```bash
    cmake ../ado && make -j8
    ```
## C++ First example
## Python First example
## More Examples

### Support Vector Machine
![Support Vector Machine applied to occupnacy data.](https://github.com/agaz1985/ado/blob/main/imgs/svm_synthetic.png?raw=true)|
