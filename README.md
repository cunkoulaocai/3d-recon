# Learning Single-View 3D Reconstruction with Limited Pose Supervision
PyTorch implementation for the paper:

[Learning Single-View 3D Reconstruction with Limited Pose Supervision](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guandao_Yang_A_Unified_Framework_ECCV_2018_paper.pdf)

### Dependencies

+ pytorch（>=0.4)
+ opencv
+ matplotlib
+ tensorboardX
+ scikit-learn

The recommended way to install the dependency is
```bash
pip install -r requirements.txt
```

## Data

Please use the following Google Drive link to download the datasets: [[drive]](https://drive.google.com/drive/folders/0B2yRgy7ZPduZMlFLSExiR0FWNzQ?usp=sharing). There are two files : `data.tar.gz` and `ShapeNetVox32.tar.gz`. Please download both of them and uncompressed it into the project root directory:
```bash
tar -xvf ShapeNetVox32.tar.gz
tar -xvf data.tar.gz
rm ShapeNetVox32.tar.gz
rm data.tar.gz
```
