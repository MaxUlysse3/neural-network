pub mod reader;

use reader::{ImageIter, Set};

fn main() {
    let iter = ImageIter::new(Set::Train);
}
