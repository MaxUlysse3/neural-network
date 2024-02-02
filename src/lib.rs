//! Library used to create and work with simple neurals networks.

use rand::Rng;
use serde::{Serialize, Deserialize};
use ndarray::{
    Array1, Array2,
};

/// The struct representing a neural network.
#[derive(Serialize, Deserialize)]
pub struct Network {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl Network {
    /// Return a new [`Network`] with no layer.
    pub fn new() -> Self {
        Self {
            weights: Vec::<_>::new(),
            biases: Vec::<_>::new(),
        }
    }
    
    /// Add a layer to the [`Network`] with random weights and biases.
    pub fn add_layer_random(&mut self, input: usize, output: usize) {
        self.weights.push(Array2::from_shape_fn((output, input), |(_, _)| <i32 as Into::<f64>>::into(rand::thread_rng().gen_range(0..1000)) / 1000f64));
        self.biases.push(Array1::from_shape_fn(output, |_| <i32 as Into<f64>>::into(rand::thread_rng().gen_range(0..1000)) / 1000f64));
    }
}
