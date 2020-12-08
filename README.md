[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/AlexanderSoroka/CNN-XRay.git/master/LICENSE.md)

# Intel landscape classification example using CNN on tf 2.x + keras

The goal of that lab is to create CNN that solves Intel Landscape Classification task

Pre-requisites:
1. TensorFlow 2.x environment

Steps to reproduce results:
1. Clone the repository:
```
git clone git@github.com:AlexanderSoroka/CNN-intel-landscape-calssification.git
```
2. Download [Intel Lanscape Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification) from kaggle to archive.zip
- unpack dataset `unzip archive.zip`
- change current directory to the folder with unpacked dataset and rename folders to make it compatible with build_image_data.py:
```
mv seg_test/seg_test val
mv seg_train/seg_train train
```

3. Generate TFRecords with build_image_data.py script:

```
python build_image_data.py --input <dataset root path> --output <tf output path>
```

Validate that total size of generated tfrecord files is close ot original dataset size

4. Run train.py to train pre-defined CNN:
```
python train.py --train '<dataset root path>/train*' --test '<dataset root path>/test*
```

5. Modify model and have fun

### [License](https://raw.githubusercontent.com/AlexanderSoroka/CNN-ArtWorks/master/LICENSE.md)

Copyright (C) 2020 Alexander Soroka.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [soroka.a.m@gmail.com](soroka.a.m@gmail.com).
