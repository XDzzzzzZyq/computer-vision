# computer-vision
 CS 207 Homeworks

## Introduction


## Image Format
This library treats images as tensors that strictly follows the shape ```(B,C,H,W)```. Where ```B``` is the batch size, 
```C``` is channel numbers, ```H,W``` are the height and width of images correspondingly.

Two storage methods are supported, unsigned bytes ranging within ```[0,255]``` or standard ```float32```.