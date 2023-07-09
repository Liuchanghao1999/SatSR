# SatSR

SatSR is a mobile edge computing (MEC)-based super-resolution (SR)-enhanced adaptive video streaming system for mobile networks with satellite backhauls. We open sourced the chunk-level simulator and DRL-based SR Scale Factor adaptation Algorithm.



### Prerequisites

- Install prerequisites (tested with Ubuntu 18.04, Tensorflow v1.1.0, TFLearn v0.3.1 and Selenium v2.39.0)



### Training

- To train a new model, put satellite and RAN link training throughout traces in `cooked_traces/sat` and `cooked_traces/user`, and corresponding validing throughout traces in `cooked_test_traces/sat` and `cooked_test_traces/user`. The trace format for simulation is `[time_stamp (sec), throughput (Mbit/sec)]`. We have put the data sample into the corresponding folder.

  

- Make sure actual video chunk files are stored in `video_server/video[1-6]`, then run

  ```
  python get_video_sizes
  ```

  

- Run this code to start training the SR scale factor adaptation algorithm.

```
python train_main.py
```

The reward signal and meta-setting of videos can be modified in `train_main.py` and `train_env.py`. 



The training process can be monitored in `sim/results/log_test` (validation) and `sim/results/log_central` (training). Tensorboard (https://www.tensorflow.org/get_started/summaries_and_tensorboard) is also used to visualize the training process, which can be invoked by running

```
python -m tensorflow.tensorboard --logdir=./results/
```

where the plot can be viewed at `localhost:6006` from a browser. 

Trained model will be saved in `results/`. 

