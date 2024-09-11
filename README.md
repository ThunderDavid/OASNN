# OASNN
This is the official repository for the paper "Brain-Inspired Online Adaptation for Remote Sensing with Spiking Neural Network."


 ### ***!We are currently organizing the code and gradually updating the repository.***


* Pretrained SNN + Adaptive Activation Scaling + Unsupervised entropy loss with temperature smooth + Confidence based instance weighting loss for detection —>  Online Adaptation for Remote Sensing
	* Starting with a pretrained SNN model, we design an efficient, unsupervised online adaptation algorithm, which adopts an approximation of the BPTT algorithm and only involves forward-in-time computation that significantly reduces the computational complexity of SNN adaptation learning.
  * we propose an adaptive activation scaling scheme to boost online SNN adaptation performance, particularly in low time-steps.
  * Furthermore, for the more challenging remote sensing detection task, we propose a confidence-based instance weighting scheme, which substantially improves adaptation performance in the detection task.
![segmentation](https://github.com/user-attachments/assets/a2d033dd-c771-44cf-ac67-6bb49efe0c42)


## Results
* State-of-the-art results on seven datasets across classification, segmentation, and detection tasks.
* To our knowledge, this work is the first to address the online adaptation of SNNs.
  
