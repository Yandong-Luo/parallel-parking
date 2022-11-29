# parallel-parking

### 配置

先装显卡驱动，装cuda，anaconda，pytorch

需要先配置这个
https://github.com/zhulf0804/PointPillars

### 运行

```
roslaunch gem_launch gem_highbay.launch
```
启动了这个之后再在另一个的终端中启动pointpillar来进行pointcloud的感知
```
rosrun pointpillars_ros pointpillars_node.py
```

