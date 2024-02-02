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

    /// Return all the activations of the layers.
    ///
    /// # Panic
    ///
    /// If the [`Network`] doesn't have any layer.
    pub fn forward(&self, input: Array1<f64>) -> Vec<Array1<f64>> {
        if self.weights.len() == 0 {
            panic!("The Network doesn't habve any layer. Try adding one with 'add_layer_random.");
        }

        let mut acts = &input;
        let mut to_return: Vec<_> = vec![];
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            //let mut z: Array1<f64> = w * acts + b;
            //z = Self::sigma(z);
            acts = &z;
            to_return.push(z);
        }

        to_return
    }

    /// The cost function (sum of squared error).
    fn cost(out: &Vec<f64>, vals: &Vec<f64>) -> f64 {
        let mut sum = 0f64;
        for (i, j) in out.iter().zip(vals) {
            sum += (j - i).powf(2f64);
        }
        sum
    }

    fn sigma(x: f64) -> f64 {
        1f64 / (1.0 - libm::exp(x))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cost() {
        assert_eq!(Network::cost(&vec![0.0, 0.1, 0.5, 1.3], &vec![0.0, 0.1, 0.5, 1.3]), 0.0);
    }
}


