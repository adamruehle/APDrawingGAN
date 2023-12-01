
# APDrawingGAN

This is a student fork of "APDrawingGAN: Generating Artistic Portrait Drawings from Face Photos with Hierarchical GANs". Our goal was to provide clear instructions for running the model locally to generate AP Drawings as well as look for potential improvements.

This project generates artistic portrait drawings from face photos using a GAN-based model.

[[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Yi_APDrawingGAN_Generating_Artistic_Portrait_Drawings_From_Face_Photos_With_Hierarchical_CVPR_2019_paper.html)
[[Original Repo]](https://github.com/yiranran/APDrawingGAN))

## Prerequisites
- Linux/macOS/Windows
- Python
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started
### Installation
- Clone the repo
```bash
$ git clone https://github.com/adamruehle/APDrawingGAN
$ cd APDrawingGAN
```
- You can install all the dependencies by
```bash
$ pip install -r requirements.txt
```

### Quick Start (Apply a Pre-trained Model)

- Download a pre-trained model (using 70 pairs in training set and augmented data) from https://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingGAN-Models1.zip (Model1) and put it in `checkpoints/formal_author`.
```bash
$ mkdir checkpoints checkpoints/formal_author
$ wget -P /checkpoints/formal_author/ https://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingGAN-Models1.zip
$ unzip /checkpoints/formal_author/APDrawingGAN-Models1.zip
```

- Then run the webapp and open the UI in your browser 
``` bash
$ python webapp.py
```
Navigate to localhost:5000 in your browser and upload/process the image there.


## Report
### Challenges Faced
- A lot of the preprocessing steps were only loosely described in the README and had to be implemented ourselves
- The align/crop preprocessing script used Matlab which is a licensed software
- The original image segmentation model was no longer available to download
- The hierarchical structure of the overall model limited what modifications were possible to the individual GANs within the model

### Summary of Improvements and Changes
- Wrote scripts to automate preprocessing based on the original author’s descriptions
- Rewrote Matlab script in Python to avoid having to obtain a Matlab license and install it, reducing install size and speeding up the alignment/cropping preprocessing from ~10 seconds to ~2 seconds
- The DeepLabV3 semantic segmentation model was used to automate the process of generating a background mask
- Created a local Python web app that handles preprocessing and generating the output to increase accessibility and useability

### Further Improvements
- Fine-tuning the DeepLabV3 model on the original testing data could be done to increase the accuracy of the masks generated
- The original paper uses separate GANs for the left eye/right eye but since the same style is used in both, they could be combined to increase the efficiency of the overall model

### Group Member Contributions
- Adam and David spent a lot of time recreating the original author’s preprocessing steps and understanding how to run and use the APDrawingGAN model locally
- Louis rewrote the Matlab script in Python
- Louis researched and implemented the DeeLabV3 semantic segmentation model
- David and Louis created the small Python web app
- Adam wrote and ran scripts to evaluate the performance of the original Matlab script vs the Python script as well as to find the overall inference time
