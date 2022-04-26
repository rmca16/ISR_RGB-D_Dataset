# ISR Tracking Dataset
The labels of ISR RGB-D Dataset were rearranged to be used as a multi-object tracking dataset, the ISR Tracking Dataset.
A unique tracking ID was associated with the same objects throughout the images, with the exception of the ''unknown'' object class that was not considered for tracking tasks. If an object disappeared or got occluded for more than 15 frames, it was considered as a new object and a new tracking ID was associated. Each image has associated a ".txt" file that contains all object labels for that image, and each object label is organized as follows: `<object class>, <tracking ID>, <bounding box center x>, <bounding box center y>, <bounding box width>, and <bounding box height>`. ISR Tracking dataset has in total 32 635 object bounding boxes and 329 object sequences.
  
#### Split
  
The ISR Tracking dataset was reorganized into two sub-datasets: ISR500 and ISR200. In the ISR500, the dataset was divided into sequences of 500 frames, which gives a total of 20 image sequences. On the other hand, the ISR200 contains 50 image sequences, which are the result of partitioning the dataset into sequences of 200 images. On both sub-datasets, the train/test image sequence split was performed by interleaving the sequences, i.e., the first sequence was used to train, the second sequence was used to test, the third sequence was used to train, and so on.
  
  
## Citation
If you use this dataset in your project or research, please consider citing it:

```
@Article{isr_tracking_dataset,
AUTHOR = {Pereira, Ricardo and Carvalho, Guilherme and Garrote, Luís and Nunes, Urbano J.},
TITLE = {{Sort and Deep-SORT Based Multi-Object Tracking for Mobile Robotics: Evaluation with New Data Association Metrics}},
JOURNAL = {Applied Sciences},
VOLUME = {12},
YEAR = {2022},
NUMBER = {3},
}
```



## Contacts
If there are any issues, you can contact us:
ricardo.pereira@isr.uc.pt
