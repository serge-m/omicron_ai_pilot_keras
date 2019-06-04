## Keras AI pilot

<p align="center">
  <img src="./images/keras-logo-2018-large-1200.png" width=40% height=40% />
</p>

### About

The ROS node, resonsible for the generating the throttle and steering based on the image.

At the moment, the input for the neural network is a single image, that we take from the Raspberry Pi camera. The output of the neural network is the values of steering and throttling.


### Feature

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that: 
- Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
- Supports both convolutional networks and recurrent networks, as well as combinations of the two.
- Runs seamlessly on CPU and GPU.

For more information:
- https://keras.io/
- https://blog.keras.io/category/tutorials.html

### How to start the node

In order to start the node, use the following commands
```
git clone git@github.com:project-omicron/ai_pilot_keras.git
cd ./ai_pilot_keras/
sudo pip3 install -r requirements.txt
roslaunch launch/ai_pilot.launch
```

### Limitations

There are no known limitations.
