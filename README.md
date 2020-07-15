# ISR RGB-D Dataset
RGB-D Dataset recorded using the ISR-InterBot mobile platform

<p align="center"><img src="assets/isr_dataset_samples.png" width="480"\></p>

## Dataset
The non-object centric RGB-D dataset was recorded in the Institute of Systems and Robotics (University of Coimbra (ISR-UC)) facilities using the Intel RealSense D435 sensor onboard the ISR-InterBot mobile platform. The dataset contains 10,000 RGB-D raw images presenting a mission performed by the ISR-InterBot platform in a lab setting (image frames represent sequence), representing the object conditions under which robotic platforms may navigate.

#### Labeling
Ten object categories (unknown, person, laptop, tvmonitor, chair, toilet, sink, desk, door-open, and door-closed) were labeled in every 4th frame achieving a total of 7,832 object-centric RGB-D images.

#### Split
The path performed during the images acquisition was approximately split by half:
	 - Training raw images ID [0-4000; 7684-9000];
   - Testing raw images ID [0-4000; 7684-9000];
   - 4271/3561 object-centric RGB-D training/testing images.


## Citation
If you use this dataset in your project or research, please consider citing it:

```
@INPROCEEDINGS{isr_rgbd_dataset,
  author={Pereira, Ricardo and Barros, Tiago and Garrote, Luís and Lopes, Ana and Nunes, Urbano J.},
  title={A Study of the Accuracy/Inference Speed Trade-Off in RGB-D Object Recognition for Mobile Robots Real-Time Applications},
  booktitle = {29th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)},
  year={2020}
}
```
