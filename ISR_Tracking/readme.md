# ISR Tracking Dataset
The labels of ISR RGB-D Dataset were rearranged to be used as a multi-object tracking dataset, the ISR Tracking Dataset.
A unique tracking ID was associated with the same objects throughout the images, with the exception of the ''unknown'' object class that was not considered for tracking tasks. If an object disappeared or got occluded for more than 15 frames, it was considered as a new object and a new tracking ID was associated. Each image has associated a ".txt" file that contains all object labels for that image, and each object label is organized as follows: <object class>, <tracking ID>, <bounding box center x>, <bounding box center y>, <bounding box width>, and <bounding box height>. ISR Tracking dataset has in total 32 635 object bounding boxes and 329 object sequences.
  
#### Split
  
  Soon....
  
  
  



## Contacts
If there are any issues, you can contact us:
ricardo.pereira@isr.uc.pt
