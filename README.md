# AI Co-pilot Bronchoscope Robot — Official PyTorch Implementation

by Jingyu Zhang, Lilu Liu, Pingyu Xiang, Qin Fang, Xiuping Nie, Honghai Ma, Jian Hu, Rong Xiong, Yue Wang, Haojian Lu.

<!-- &#x26A0; **More details of this repository are COMING SOON!** -->

## Introduction
Bronchoscopy is a critical diagnostic and therapeutic tool in managing lung diseases. Nevertheless, the unequal distribution of medical resources confines access to bronchoscopy primarily to well-equipped hospitals in developed regions. The scarcity of experienced practitioners contributes to the unavailability of bronchoscopic services in underdeveloped areas. In response, we present an artificial intelligence (AI) co-pilot bronchoscope robot that empowers novice doctors to conduct lung examinations as safely and adeptly as experienced colleagues. The system features a user-friendly, plug-and-play catheter with diameters of 3.3 mm or 2.1 mm, devised for robot-assisted steering, facilitating access to bronchi beyond the fifth generation in average adult patients. Drawing upon historical bronchoscopic videos and expert imitation, our AI-human shared control algorithm enables novice doctors to attain safe steering along the central axis of the bronchus, mitigating misoperations and reducing contact between the bronchoscope and bronchial walls. To evaluate the system’s reliable steering performance, we conducted in vitro assessments using realistic human bronchial phantoms simulating human respiratory behavior, and preclinical in vivo tests in live porcine lungs. Both in vitro and in vivo results underscore that our system equips novice doctors with the skills to perform lung examinations as expertly as seasoned practitioners. This study carries the potential to not only augment the existing bronchoscopy medical paradigm but also provide innovative strategies to tackle the pressing issue of medical resource disparities through AI assistance.

<!-- <img src="figs/overview.jpg#pic_left" alt="avatar" style="zoom:30%;" /> -->
<img src="figs/overview.jpg#pic_left" alt="avatar" style="zoom:40%;" />


## Usage

### Prerequisites
* Python 3.6.2
* PyTorch 1.9.1 and torchvison (https://pytorch.org/)
* VTK 8.2.0
* Pyrender 0.1.45
* PyBullet 3.2.2
* CUDA 10.2


### Installation
<!-- * PyTorch >= 1.6
* SimpleITK
* OpenCV
* SciPy
* Numpy -->
Necessary Python packages can be installed by

```bash
pip install -r requirements.txt
```

### Data

### Train
```
> python train.py  --dataset-dir YOUR_AIRWAY_AND_CENTERLINE_DIR
```

### Test
```
> python test.py --val-dataset YOUR_AIRWAY_AND_CENTERLINE_DIR  --model-dir YOUR_CHECK_POINT_DIR ----load-epoch EPOCH_COUNT
```

Visualization and interaction GUI in PyBullet:
<img src="figs/sim.png#pic_left" alt="avatar" style="zoom:40%;" />
User provide discrete huamn commands (left, right, up, down, forward) to control the simulated bronchocope robot to safely instert in bronchus.

## Results

### Image translation (structure-preserving domain adaptation)
<img src="figs/DA.png#pic_left" alt="avatar" style="zoom:100%;" />

### Simulation
<img src="figs/simulation1.png#pic_left" alt="avatar" style="zoom:100%;" /> <img src="figs/simulation2.png#pic_left" alt="avatar" style="zoom:100%;" />

### In-vitro
<img src="figs/invirtro.png#pic_left" alt="avatar" style="zoom:100%;" />
### In-vivo
<img src="figs/invivo.png#pic_left" alt="avatar" style="zoom:100%;" />
