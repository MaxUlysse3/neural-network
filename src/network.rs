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
            panic!("The Network doesn't have any layer.");
        }

        let mut acts = &input;
        let mut to_return: Vec<_> = vec![];
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            // eprintln!("{:?}", acts.dim());
            // eprintln!("{:?}", (w.dot(acts)).dim());
            let mut z = w.dot(acts) + b;
            z.iter_mut().for_each(|x| *x = Self::sigma(*x));
            to_return.push(z);
            acts = to_return.last().unwrap();
        }
        to_return.insert(0, input);

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

    /// Compute the gradients for one (input, output) pair.
    pub fn backpropagate(&self, input: Array1<f64>, expected: Array1<f64>) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
        // Initialize the delta
        // let mut delta = Vec::<Array1<f64>>::new();

        // Initialize the gradient
        let mut gradient_w = Vec::<Array2<f64>>::new();
        let mut gradient_b = Vec::<Array1<f64>>::new();

        // Get the activations of all layers
        let mut acts = self.forward(input).into_iter().enumerate().rev();
        // println!("{:?}", acts.len());

        let last_act = acts.next().unwrap().1;
        let mut last_delta = Array1::<f64>::from_shape_fn(last_act.raw_dim(), |x| (last_act[x] - expected[x]) * last_act[x] * (1.0 - last_act[x]));

        for (idx, act) in acts {
            gradient_w.push(Array2::<f64>::from_shape_fn((last_delta.len(), act.len()), |(i, j)| last_delta[i] * act[j]));
            gradient_b.push(last_delta.clone());

            // eprintln!("{:?}", idx + 1);
            last_delta = self.weights[idx].t().dot(&last_delta).map(|x| Self::sigma_prime(*x));
        }
        // println!("{:?}", gradient_w.len());

        (gradient_w, gradient_b)

        
    }

    /// Find the gradients for multiple (inputs, outputs) pair.
    pub fn find_gradient<I>(&self, iterator: &mut I, num_iter: usize) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) where
        I: Iterator<Item = (Array1<f64>, Array1<f64>)> {
        let (input, target) = iterator.next().unwrap();
        let (mut gradient_w, mut gradient_b) = self.backpropagate(input, target);
        for (_, (input, target)) in (0..num_iter-1).zip(iterator) {
            let (gws, gbs) = self.backpropagate(input, target);
            // println!("{:#?}, {:#?}", gws.len(), gbs.len());
            for (i, gw) in gradient_w.iter_mut().enumerate() {
                // println!("{:?}, {:?}", gw.raw_dim(), gws[i].raw_dim());
                gw.iter_mut().zip(gws[i].iter()).for_each(|(x, y)| *x += y);
            }
            for (i, gb) in gradient_b.iter_mut().enumerate() {
                gb.iter_mut().zip(gbs[i].iter()).for_each(|(x, y)| *x += y);
            }
        }
        for (gws, gbs) in (gradient_w, gradient_b) {
            // TODO
        }
        (gradient_w, gradient_b)
    }

    pub fn learn<I>(&mut self, iterator: &mut I, num_iter: usize, learn_num: usize) {
        
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

        let nb = NetworkBuilder::new(w0.shape()[0]).add_layer(w0.clone(), b0.clone())
                                                          .add_layer(w1.clone(), b1.clone());
        let n = nb.build()?;

        assert_eq!(n.weights(), &vec![w0, w1]);
        assert_eq!(n.biases(), &vec![b0, b1]);

        Ok(())
    }

    mod cut_square {
        use ndarray::{Array2, Array1};
        use rand::Rng;

        const SLOPE: f64 = 2.0;
        const H: f64 = 0.3;

        fn predicate(x: f64, y: f64) -> bool {
            let line = |x: f64| SLOPE * x + H;

            line(x) > y
        }

        pub fn gen_vals() -> (Vec<f64>, Vec<f64>) {
            let x = rand::thread_rng().gen_range(0..1000) as f64 / 1000f64;
            let y = rand::thread_rng().gen_range(0..1000) as f64 / 1000f64;

            let out = match predicate(x, y) {
                true => vec![1.0, 0.0],
                false => vec![0.0, 1.0],
            };

            (vec![x, y], out)
        }

        pub fn gen_network() -> super::Network {
            super::NetworkBuilder::new(2).add_layer_random(2).add_layer_random(2).build().unwrap()
        }

        pub fn cost(network: &super::Network) -> f64 {
            let mut cost = 0.0;
            let n = 10000;

            for _ in 0..n {
                let (input, target) = gen_vals();
                let out = network.forward(input.into()).last().unwrap().to_vec();
                cost += super::Network::cost(&out, &target)
            }

            cost / n as f64
        }

        pub fn gradient(network: &super::Network) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
            let (input, target) = gen_vals();
            let (mut gradient_w, mut gradient_b) = network.backpropagate(input.into(), target.into());
            // println!("{:?}", gradient_w.len());
            let n = 10000;

            for _ in 0..n {
                let (input, target) = gen_vals();
                let (new_gradient_w, new_gradient_b) = network.backpropagate(input.into(), target.into());
                for (gw, ngw) in gradient_w.iter_mut().zip(new_gradient_w.iter()) {
                    *gw += ngw;
                }

                for (gb, ngb) in gradient_b.iter_mut().zip(new_gradient_b.iter()) {
                    *gb += ngb;
                }
            }

            for gw in gradient_w.iter_mut() {
                gw.iter_mut().for_each(|x| *x = *x / n as f64);
            }
            for gb in gradient_b.iter_mut() {
                gb.iter_mut().for_each(|x| *x = *x / n as f64);
            }

            (gradient_w, gradient_b)
        }

    }

    // #[test]
    // fn test_gradient() {
    //     let mut n = cut_square::gen_network();

    //     let c1 = cut_square::cost(&n);

    //     let (gradient_w, gradient_b) = cut_square::gradient(&n);

    //     let delta_m = 0.1;
    //     for (w, b) in n.weights.iter_mut().zip(n.biases.iter_mut()) {
    //         *w = w.clone() + Array2::from_elem(w.raw_dim(), &delta_m);
    //         *b = b.clone() + Array1::from_elem(b.raw_dim(), &delta_m);
    //     }

    //     let c2 = cut_square::cost(&n);

    //     let mut product = 0.0;

    //     gradient_w.into_iter()
    //                 .zip(gradient_b.into_iter())
    //                 .for_each(|(w, b)| {
    //                     w.into_iter().for_each(|weight| product += weight * delta_m);
    //                     b.into_iter().for_each(|bias| product += bias * delta_m);
    //                 });

    //     println!("{:?}, {:?}", c2 - c1, product);
    //     panic!("");
    // }

    #[test]
    fn test_gradient() {
        let mut n = cut_square::gen_network();

        let c1 = cut_square::cost(&n);

        let (gradient_w, gradient_b) = cut_square::gradient(&n);

        // println!("{:?}", gradient_w.len());

        for (idx, (w, b)) in n.weights.iter_mut().zip(n.biases.iter_mut()).enumerate() {
            *w = &*w - &gradient_w[idx];
            *b = &*b - &gradient_b[idx];
        }

        let c2 = cut_square::cost(&n);

        // println!("{:?} -> {:?}", c1, c2 - c1);
        assert!(c2 - c1 < 0f64);
    }
}
