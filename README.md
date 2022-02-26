# 基于Pytorch的MNIST手写数字识别

## 基本介绍

本项目是基于pytorch的MNIST手写数字识别，模型采用卷积神经网络。

## 运行环境

操作系统=windows10

显卡=GeForce GTX 1050

python=3.8.8

conda=4.10.3

cuda=10.2

pytorch=1.10.1+cu102

torchvision=0.11.2+cu102

## 主要文件

Handwritten Digit Recognition

- conf.py：设置模型的超参数
- model.py：设置模型的结构
- get_dataset.py：包含数据集类
- train_and_test.py：包含模型的训练和测试函数
- main.py：主函数，执行模型的训练和测试
- data：存放MNIST数据集
- models：存放训练好的模型

.py文件之间的依赖关系如下：

![依赖](https://gitee.com/LianqingHikari/my-blog-hub/raw/master/202202261859076.png)

## 运行

直接执行main.py即可。

## 运行结果

在batch_size=128时训练100个epochs得到如下结果：

```python
[1/100] |Train Loss: 0.02832443, Acc: 0.6100 |Val Loss: 0.00369474, Acc: 0.8627
[2/100] |Train Loss: 0.00389740, Acc: 0.8428 |Val Loss: 0.00331395, Acc: 0.8751
[3/100] |Train Loss: 0.00364885, Acc: 0.8585 |Val Loss: 0.00350785, Acc: 0.8697
[4/100] |Train Loss: 0.00358981, Acc: 0.8677 |Val Loss: 0.00346696, Acc: 0.8773
[5/100] |Train Loss: 0.00360772, Acc: 0.8732 |Val Loss: 0.00355460, Acc: 0.8701
[6/100] |Train Loss: 0.00362383, Acc: 0.8748 |Val Loss: 0.00375740, Acc: 0.8743
[7/100] |Train Loss: 0.00378970, Acc: 0.8777 |Val Loss: 0.00429138, Acc: 0.8530
[8/100] |Train Loss: 0.00374522, Acc: 0.8814 |Val Loss: 0.00306524, Acc: 0.9033
[9/100] |Train Loss: 0.00361992, Acc: 0.8846 |Val Loss: 0.00371016, Acc: 0.8844
[10/100] |Train Loss: 0.00315170, Acc: 0.8971 |Val Loss: 0.00362909, Acc: 0.8892
[11/100] |Train Loss: 0.00340556, Acc: 0.8933 |Val Loss: 0.00309910, Acc: 0.8991
[12/100] |Train Loss: 0.00341472, Acc: 0.8945 |Val Loss: 0.00263950, Acc: 0.9191
[13/100] |Train Loss: 0.00286040, Acc: 0.9069 |Val Loss: 0.00315174, Acc: 0.8995
[14/100] |Train Loss: 0.00284958, Acc: 0.9048 |Val Loss: 0.00386439, Acc: 0.8738
[15/100] |Train Loss: 0.00279225, Acc: 0.9083 |Val Loss: 0.00339151, Acc: 0.8952
[16/100] |Train Loss: 0.00225905, Acc: 0.9238 |Val Loss: 0.00259584, Acc: 0.9135
[17/100] |Train Loss: 0.00209337, Acc: 0.9266 |Val Loss: 0.00252821, Acc: 0.9187
[18/100] |Train Loss: 0.00205188, Acc: 0.9282 |Val Loss: 0.00182875, Acc: 0.9382
[19/100] |Train Loss: 0.00209296, Acc: 0.9296 |Val Loss: 0.00174238, Acc: 0.9457
[20/100] |Train Loss: 0.00162558, Acc: 0.9411 |Val Loss: 0.00225335, Acc: 0.9276
[21/100] |Train Loss: 0.00142792, Acc: 0.9490 |Val Loss: 0.00187212, Acc: 0.9405
[22/100] |Train Loss: 0.00135009, Acc: 0.9511 |Val Loss: 0.00171346, Acc: 0.9419
[23/100] |Train Loss: 0.00121842, Acc: 0.9557 |Val Loss: 0.00135722, Acc: 0.9530
[24/100] |Train Loss: 0.00111837, Acc: 0.9587 |Val Loss: 0.00157516, Acc: 0.9467
[25/100] |Train Loss: 0.00110685, Acc: 0.9599 |Val Loss: 0.00100404, Acc: 0.9677
[26/100] |Train Loss: 0.00113835, Acc: 0.9585 |Val Loss: 0.00271294, Acc: 0.9138
[27/100] |Train Loss: 0.00095891, Acc: 0.9646 |Val Loss: 0.00122083, Acc: 0.9609
[28/100] |Train Loss: 0.00093043, Acc: 0.9655 |Val Loss: 0.00190769, Acc: 0.9372
[29/100] |Train Loss: 0.00094188, Acc: 0.9646 |Val Loss: 0.00124967, Acc: 0.9566
[30/100] |Train Loss: 0.00090829, Acc: 0.9675 |Val Loss: 0.00109396, Acc: 0.9643
[31/100] |Train Loss: 0.00077537, Acc: 0.9715 |Val Loss: 0.00107043, Acc: 0.9638
[32/100] |Train Loss: 0.00076886, Acc: 0.9718 |Val Loss: 0.00128136, Acc: 0.9605
[33/100] |Train Loss: 0.00067405, Acc: 0.9749 |Val Loss: 0.00115335, Acc: 0.9642
[34/100] |Train Loss: 0.00056755, Acc: 0.9788 |Val Loss: 0.00199854, Acc: 0.9390
[35/100] |Train Loss: 0.00061226, Acc: 0.9766 |Val Loss: 0.00111622, Acc: 0.9676
[36/100] |Train Loss: 0.00060089, Acc: 0.9764 |Val Loss: 0.00085925, Acc: 0.9741
[37/100] |Train Loss: 0.00060667, Acc: 0.9768 |Val Loss: 0.00153564, Acc: 0.9563
[38/100] |Train Loss: 0.00055789, Acc: 0.9789 |Val Loss: 0.00133327, Acc: 0.9593
[39/100] |Train Loss: 0.00051757, Acc: 0.9799 |Val Loss: 0.00109348, Acc: 0.9673
[40/100] |Train Loss: 0.00051699, Acc: 0.9800 |Val Loss: 0.00105077, Acc: 0.9688
[41/100] |Train Loss: 0.00050937, Acc: 0.9801 |Val Loss: 0.00132884, Acc: 0.9629
[42/100] |Train Loss: 0.00052837, Acc: 0.9800 |Val Loss: 0.00120465, Acc: 0.9662
[43/100] |Train Loss: 0.00062531, Acc: 0.9772 |Val Loss: 0.00214554, Acc: 0.9486
[44/100] |Train Loss: 0.00058773, Acc: 0.9786 |Val Loss: 0.00087118, Acc: 0.9765
[45/100] |Train Loss: 0.00031875, Acc: 0.9874 |Val Loss: 0.00117240, Acc: 0.9671
[46/100] |Train Loss: 0.00036127, Acc: 0.9856 |Val Loss: 0.00101660, Acc: 0.9717
[47/100] |Train Loss: 0.00034309, Acc: 0.9865 |Val Loss: 0.00082599, Acc: 0.9771
[48/100] |Train Loss: 0.00024860, Acc: 0.9898 |Val Loss: 0.00097216, Acc: 0.9737
[49/100] |Train Loss: 0.00032219, Acc: 0.9870 |Val Loss: 0.00081312, Acc: 0.9779
[50/100] |Train Loss: 0.00030755, Acc: 0.9877 |Val Loss: 0.00142321, Acc: 0.9617
[51/100] |Train Loss: 0.00034559, Acc: 0.9862 |Val Loss: 0.00103073, Acc: 0.9740
[52/100] |Train Loss: 0.00023133, Acc: 0.9902 |Val Loss: 0.00101282, Acc: 0.9763
[53/100] |Train Loss: 0.00024078, Acc: 0.9898 |Val Loss: 0.00093933, Acc: 0.9773
[54/100] |Train Loss: 0.00024489, Acc: 0.9901 |Val Loss: 0.00081498, Acc: 0.9788
[55/100] |Train Loss: 0.00020370, Acc: 0.9914 |Val Loss: 0.00106013, Acc: 0.9719
[56/100] |Train Loss: 0.00023469, Acc: 0.9906 |Val Loss: 0.00103815, Acc: 0.9750
[57/100] |Train Loss: 0.00018428, Acc: 0.9923 |Val Loss: 0.00120362, Acc: 0.9682
[58/100] |Train Loss: 0.00027145, Acc: 0.9885 |Val Loss: 0.00144152, Acc: 0.9666
[59/100] |Train Loss: 0.00019652, Acc: 0.9924 |Val Loss: 0.00116154, Acc: 0.9732
[60/100] |Train Loss: 0.00017198, Acc: 0.9923 |Val Loss: 0.00147633, Acc: 0.9668
[61/100] |Train Loss: 0.00014546, Acc: 0.9938 |Val Loss: 0.00113453, Acc: 0.9746
[62/100] |Train Loss: 0.00016329, Acc: 0.9934 |Val Loss: 0.00107619, Acc: 0.9740
[63/100] |Train Loss: 0.00014886, Acc: 0.9938 |Val Loss: 0.00095467, Acc: 0.9779
[64/100] |Train Loss: 0.00017457, Acc: 0.9921 |Val Loss: 0.00117823, Acc: 0.9719
[65/100] |Train Loss: 0.00011755, Acc: 0.9946 |Val Loss: 0.00124778, Acc: 0.9705
[66/100] |Train Loss: 0.00019413, Acc: 0.9924 |Val Loss: 0.00120264, Acc: 0.9720
[67/100] |Train Loss: 0.00016409, Acc: 0.9934 |Val Loss: 0.00140696, Acc: 0.9705
[68/100] |Train Loss: 0.00015440, Acc: 0.9937 |Val Loss: 0.00092666, Acc: 0.9790
[69/100] |Train Loss: 0.00007453, Acc: 0.9966 |Val Loss: 0.00112604, Acc: 0.9756
[70/100] |Train Loss: 0.00008966, Acc: 0.9963 |Val Loss: 0.00114503, Acc: 0.9764
[71/100] |Train Loss: 0.00010428, Acc: 0.9954 |Val Loss: 0.00104472, Acc: 0.9769
[72/100] |Train Loss: 0.00020612, Acc: 0.9920 |Val Loss: 0.00106537, Acc: 0.9765
[73/100] |Train Loss: 0.00011338, Acc: 0.9953 |Val Loss: 0.00091613, Acc: 0.9805
[74/100] |Train Loss: 0.00007337, Acc: 0.9966 |Val Loss: 0.00094644, Acc: 0.9797
[75/100] |Train Loss: 0.00006202, Acc: 0.9971 |Val Loss: 0.00124200, Acc: 0.9738
[76/100] |Train Loss: 0.00006080, Acc: 0.9972 |Val Loss: 0.00095912, Acc: 0.9796
[77/100] |Train Loss: 0.00007308, Acc: 0.9963 |Val Loss: 0.00120112, Acc: 0.9747
[78/100] |Train Loss: 0.00013090, Acc: 0.9945 |Val Loss: 0.00103017, Acc: 0.9790
[79/100] |Train Loss: 0.00007862, Acc: 0.9962 |Val Loss: 0.00117800, Acc: 0.9770
[80/100] |Train Loss: 0.00005346, Acc: 0.9973 |Val Loss: 0.00097963, Acc: 0.9807
[81/100] |Train Loss: 0.00002003, Acc: 0.9993 |Val Loss: 0.00112380, Acc: 0.9769
[82/100] |Train Loss: 0.00004203, Acc: 0.9982 |Val Loss: 0.00108346, Acc: 0.9776
[83/100] |Train Loss: 0.00011715, Acc: 0.9950 |Val Loss: 0.00143138, Acc: 0.9722
[84/100] |Train Loss: 0.00008445, Acc: 0.9964 |Val Loss: 0.00137907, Acc: 0.9736
[85/100] |Train Loss: 0.00005927, Acc: 0.9976 |Val Loss: 0.00110429, Acc: 0.9783
[86/100] |Train Loss: 0.00002503, Acc: 0.9988 |Val Loss: 0.00103319, Acc: 0.9797
[87/100] |Train Loss: 0.00004329, Acc: 0.9980 |Val Loss: 0.00102649, Acc: 0.9800
[88/100] |Train Loss: 0.00008076, Acc: 0.9969 |Val Loss: 0.00119290, Acc: 0.9788
[89/100] |Train Loss: 0.00002316, Acc: 0.9988 |Val Loss: 0.00109010, Acc: 0.9800
[90/100] |Train Loss: 0.00006130, Acc: 0.9973 |Val Loss: 0.00141910, Acc: 0.9747
[91/100] |Train Loss: 0.00009657, Acc: 0.9959 |Val Loss: 0.00117967, Acc: 0.9771
[92/100] |Train Loss: 0.00003580, Acc: 0.9983 |Val Loss: 0.00122260, Acc: 0.9764
[93/100] |Train Loss: 0.00001791, Acc: 0.9993 |Val Loss: 0.00104773, Acc: 0.9802
[94/100] |Train Loss: 0.00000358, Acc: 0.9999 |Val Loss: 0.00103321, Acc: 0.9806
[95/100] |Train Loss: 0.00000467, Acc: 0.9998 |Val Loss: 0.00100076, Acc: 0.9805
[96/100] |Train Loss: 0.00000111, Acc: 1.0000 |Val Loss: 0.00101395, Acc: 0.9803
[97/100] |Train Loss: 0.00000084, Acc: 1.0000 |Val Loss: 0.00099476, Acc: 0.9812
[98/100] |Train Loss: 0.00000069, Acc: 1.0000 |Val Loss: 0.00101691, Acc: 0.9810
[99/100] |Train Loss: 0.00000063, Acc: 1.0000 |Val Loss: 0.00102176, Acc: 0.9812
[100/100] |Train Loss: 0.00000069, Acc: 1.0000 |Val Loss: 0.00099358, Acc: 0.9811
Test Acc:0.981800
```

损失曲线如下：

![loss](https://gitee.com/LianqingHikari/my-blog-hub/raw/master/202202261930365.png)

准确率曲线如下：

![Accuracy](https://gitee.com/LianqingHikari/my-blog-hub/raw/master/202202261930728.png)
