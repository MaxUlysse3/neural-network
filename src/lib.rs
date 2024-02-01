//! Library used to create and work with simple neurals networks.

use rand::Rng;
use serde::{Serialize, Deserialize};
use rulinalg::{
    matrix::Matrix,
    vector::Vector,
};

pub struct Network {
    weights: Vec<Matrix<f64>>,
    biases: Vec<Vector<f64>>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            weights: Vec::<_>::new(),
            biases: Vec::<_>::new(),
        }
    }
    
    pub fn add_layer_random(&mut self, input: usize, output: usize) {
        self.weights.push(Matrix::from_fn(output, input, |_, _| <i32 as Into::<f64>>::into(rand::thread_rng().gen_range(0..1000)) / 1000f64));
    }
}
