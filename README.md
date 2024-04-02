# Neural Network

A simple implementation of the multi-level perceptron in Rust.
Essentially backward propagation and stochastic gradient descent.

# Installing

First install [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).
Then `git clone https://github.com/MaxUlysse3/neural-network.git`.

## Mnist database

You'll need to download the [mnist database](http://yann.lecun.com/exdb/mnist/).
Extract the files, place them and rename them like so:
```
neural-network/mnist/test-images.idx
neural-network/mnist/test-labels.idx
neural-network/mnist/train-images.idx
neural-network/mnist/train-labels.idx
```

Finally, enter the neural-network folder and `cargo run`.
