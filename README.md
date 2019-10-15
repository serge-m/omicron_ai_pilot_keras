# ai_pilot_keras

ai_pilot_keras is a ROS node responsible for decision making and trajectory planning of [robocar](https://github.com/serge-m/omicron_robocar).
It uses deep convolutional neural network to generate throttle and steering values from input images.

At the moment, the input for the neural network is a single image, that we take from the Raspberry Pi camera. The output of the neural network is the values of steering and throttling.


## Implementation

The neural network is implemented using [Keras](https://keras.io/) framework.

See also: [keras tutorials](https://blog.keras.io/category/tutorials.html).


## Usage

See [project wiki](https://github.com/serge-m/omicron_ai_pilot_keras/wiki) for usage examples and documentation.
