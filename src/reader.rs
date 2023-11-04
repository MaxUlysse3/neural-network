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
        Self {
            image,
            label,
        }
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
            Set::Train => "train",
            Set::Test => "test",
        };
        let mut to_return = Self {
            data_img: fs::read(format!("../mnist/{}-{}.idx", set_str, "images")).expect("File not found.").into_iter(),
            data_lab: fs::read(format!("../mnist/{}-{}.idx", set_str, "labels")).expect("File not found.").into_iter(),
        };
        
        println!("{:?}", to_return.data_img.next_chunk::<16>().unwrap());
        println!("{:?}", to_return.data_lab.next_chunk::<8>().unwrap());
        to_return
    }
}

impl Iterator for ImageIter {
    type Item = Image;

    fn next(&mut self) -> Option<Self::Item> {
        match self.data_img.next_chunk::<784>() {
            Err(_) => None,
            Ok(v) => match self.data_lab.next() {
                None => None,
                Some(l) => Some(Image::new(v.into(), l))
            }
        }
    }
}
