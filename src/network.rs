use rand::Rng;
use serde::{Serialize, Deserialize};
use ndarray::{
    Array1, Array2,
};

/// The struct representing a neural network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl Network {
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
            eprintln!("{:?}", acts.dim());
            eprintln!("{:?}", (w.dot(acts)).dim());
            let mut z = w.dot(acts) + b;
            z.iter_mut().for_each(|x| *x = Self::sigma(*x));
            to_return.push(z);
            acts = to_return.last().unwrap();
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

    pub fn backpropagate(&mut self, input: Array1<f64>) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
        // Initialize the gradiant
        let mut nabla_w = Vec::<Array2<f64>>::new();
        let mut nabla_b = Vec::<Array1<f64>>::new();

        // Get the activations of all layers
        let acts = self.forward(input);

        for (weights, biases) in self.weights.iter().rev().zip(self.biases.iter().rev()) {
            
        }
        // TODO
    }

    fn sigma(x: f64) -> f64 {
        1f64 / (1.0 + libm::exp(-x))
    }

    fn sigma_prime(x: f64) -> f64 {
        Self::sigma(x) * (1.0 - Self::sigma(x))
    }
}

/// Getters for ['Network'].
impl Network {
    /// Return the weight matrices.
    pub fn weights(&self) -> &Vec<Array2<f64>> {
        &self.weights
    }

    /// Return the biase vectors.
    pub fn biases(&self) -> &Vec<Array1<f64>> {
        &self.biases
    }
}

/// A builder used to create a new [`Network`].
#[derive(Debug, Clone)]
pub struct NetworkBuilder {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    output_size: usize,
}

impl NetworkBuilder {
    /// Return a new [`NetworkBuilder`] with no layer.
    pub fn new(input_size: usize) -> Self {
        Self {
            weights: Vec::<_>::new(),
            biases: Vec::<_>::new(),
            output_size: input_size,
        }
    }
    
    /// Add a layer to the [`NetworkBuilder`] with random weights and biases.
    pub fn add_layer_random(mut self, size: usize) -> Self {
        self.weights.push(Array2::from_shape_fn((size, self.output_size), |(_, _)| <i32 as Into::<f64>>::into(rand::thread_rng().gen_range(0..1000)) / 1000f64));
        self.biases.push(Array1::from_shape_fn(size, |_| <i32 as Into<f64>>::into(rand::thread_rng().gen_range(0..1000)) / 1000f64));
        self.output_size = size;
        self
    }

    /// Add a layer to the [`NetworkBuilder`] with given weights and biases.
    pub fn add_layer(mut self, weights: Array2<f64>, biases: Array1<f64>) -> Self {
        self.weights.push(weights);
        self.biases.push(biases);
        self
    }

    /// Return a [`Network`] with set layers and check compatibility between layer sizes.
    ///
    /// # Error
    ///
    /// Two consecutive layers `(w0, b0)` and `(w1, b1)` need to verify `w0.dim().0 == b0.dim()`
    /// and `b0.dim() == w1.dim().1`
    pub fn build(self) -> Result<Network, (Self, String)> {
        // Check if there's at least one layer.
        match self.weights.len() {
            0 => return Err((self, "A 'Network' must have at least 1 layer. This has 0.".into())),
            _ => (),
        }

        // Check layer dimensions compatibility.
        let mut last_dim = self.weights[0].dim().1;
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            match (w.dim().0 == b.dim()) && (w.dim().1 == last_dim) {
                false => return Err((self, "Layer dimensions did not match.".into())),
                true => last_dim = b.dim(),
            }
        }

        Ok(Network {
            weights: self.weights,
            biases: self.biases,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_cost() {
        assert_eq!(Network::cost(&vec![0.0, 0.1, 0.5, 1.3], &vec![0.0, 0.1, 0.5, 1.3]), 0.0);
    }

    #[test]
    fn test_new() -> Result<(), (NetworkBuilder, String)>{
        let w0 = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
        let b0 = arr1(&[2.0, -1.0]);

        let w1 = arr2(&[[1.0, 1.0], [-1.0, -1.0], [2.0, -2.0]]);
        let b1 = arr1(&[0.0, -1.0, 1.0]);

        let nb = NetworkBuilder::new().add_layer(w0.clone(), b0.clone())
                                                          .add_layer(w1.clone(), b1.clone());
        let n = nb.build()?;

        assert_eq!(n.weights(), &vec![w0, w1]);
        assert_eq!(n.biases(), &vec![b0, b1]);

        Ok(())
    }

    #[test]
    #[should_panic(expected = "Layer dimensions did not match")]
    fn test_new_wrong_dimensions() {
        let nb = NetworkBuilder::new().add_layer_random(3, 2).add_layer_random(3, 2);
        nb.build().unwrap();
    }
}
