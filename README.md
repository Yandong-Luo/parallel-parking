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
重新规划路径，用这个
```
rosrun hybrid_astar replanning
```
当当前规划的路径看起来不错，执行以下指令完成对path的选择，然后车辆根据pid进行运动
```
rosrun hybrid_astar selectpath
```
