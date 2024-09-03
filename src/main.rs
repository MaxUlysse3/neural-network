#![feature(iter_next_chunk)]

pub mod reader;

use neural_network::network::{Network, NetworkBuilder};
use reader::{ImageIter, Set, Image};

fn main() {
    let mut iter = ImageIter::new(Set::Train);

    let mut n = NetworkBuilder::new(784).add_layer_random(16).add_layer_random(16).add_layer_random(10);
    let n = n.build().unwrap();
    // n.backpropagate(iter.next().unwrap().into(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].into());
    n.find_gradient(&mut iter.map(|i| i.into()), 2);
}
