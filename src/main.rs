#![feature(iter_next_chunk)]

pub mod reader;

use neural_network::Network;
use reader::{ImageIter, Set, Image};

fn main() {
    let mut iter = ImageIter::new(Set::Train);

    iter.nth(37).unwrap().show();

    let n = Network::new().add_layer_random(10, 16);
}
