# DNN-NOMA
Simulation for "[Deep Neural Network-Based Active User Detection for Grant-Free NOMA Systems](https://ieeexplore.ieee.org/document/8968401)".  
Large scale fading is ignored in this repository.
## Requirements
* Numpy
* Tensorflow
* Ray
## Run
* Run main to generate data and train the network
## Old versions
* AIO <br>
(Could wait for days)
  * main_0: Many Codebooks.
  * main_0_yield: Generator and very very very slow.
  * main_Ray0: Ray toolkit with codebook check.
  * main_Ray1: Ray toolkit with generator and threadsafe decorator, still slow.
* Seperate<br>
 (Could wait for 30 minutes for 2400000 datasets generation)
  * main: Run the code.
  * filemanager: To be finished. Save weights and codebook to deployment.
  * generator: Functions.
<br>
If there is any problem please email leo01412123@gmail.com  
