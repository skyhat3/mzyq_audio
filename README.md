seg.py文件包含了对原始数据的处理，运行后会在目标文件夹生成分割音频和csv文件

trainer.py定义了训练的trainer，并且给出了一个简单的实例化例子，可以直接运行

utils.py定义了一些用到的工具函数和一些定义的Datasets类，直接运行会生成csv文件，包含乐器ID和乐器名、文件名、标签等；还可以进一步处理生成npy文件

数据集源于CCMUSIC:https://ccmusic-database.github.io/database/csmtd.html
