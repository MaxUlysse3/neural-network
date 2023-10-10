use std::{
    fs,
    iter,
};

use flo_draw::*;
use flo_draw::{canvas};

pub struct Image {
    image: Vec<u8>,
    label: u8,
}

impl Image {
    fn new(image: Vec<u8>, label: u8) -> Self {
        image,
        label,
    }

    #[inline]
    pub fn image(self) -> Vec<u8> {
        self.image
    }

    #[inline]
    pub fn label(&self) -> u8 {
        self.label
    }
}

impl From<Image> for (Vec<u8>, u8) {
    #[inline]
    fn from(value: Image) -> Self {
        (value.image, value.label)
    }
}

impl From<Image> for Vec<u8> {
    #[inline]
    fn from(value: Image) -> Self {
        value.image()
    }
}

pub enum Set {
    Train,
    Test,
}

const PATH: &str = "../mnist/{}-{}.idx";

type VecIntoIter<T> = <Vec<T> as IntoIterator>::IntoIter;
pub struct ImageIter {
    data_img: VecIntoIter<u8>,
    data_lab: VecIntoIter<u8>,
}

impl ImageIter {
    pub fn new(set: Set) -> Self {
        let set_str = match set {
            Train => "train",
            Test => "test",
        };
        Self {
            data_img: fs::read(format!("../mnist/{}-{}.idx", set_str, "images")).expect("File not found.").into_iter(),
            data_lab: fs::read(format!("../mnist/{}-{}.idx", set_str, "labels")).expect("File not found.").into_iter(),
        }
    }
}
