#![feature(iter_next_chunk)]

pub mod reader;

use neural_network::network::{Network, NetworkBuilder};
use reader::{ImageIter, Set, Image};

fn main() {
    let mut iter = ImageIter::new(Set::Train);

    iter.nth(36).unwrap().show();

    let mut n = NetworkBuilder::new(10).add_layer_random(16);
    let n = n.build().unwrap();
    n.forward(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].into());
}
