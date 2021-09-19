#![no_std]

extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

use alloc::{string::String, vec::Vec};
pub use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};

#[derive(Debug, Clone, PartialEq, ProcBlock)]
#[transform(input = [u8; _], output = [i32; _])]
pub struct ByteInputConversion {}

impl ByteInputConversion {
    pub const fn new() -> Self {
        ByteInputConversion {}
    }
}

impl Default for ByteInputConversion {
    fn default() -> Self {
        ByteInputConversion::new()
    }
}

impl Transform<Tensor<u8>> for ByteInputConversion {
    type Output = Tensor<i32>;

    fn transform(&mut self, input: Tensor<u8>) -> Self::Output {
        let mut json = String::new();

        for element in input.elements() {
            json.push(*element as char);
        }

        let vec: Vec<i32> = serde_json_core::from_str(&json).unwrap();

        Tensor::new_vector(vec)
    }
}

impl HasOutputs for ByteInputConversion {
    fn set_output_dimensions(&mut self, dimensions: &[usize]) {
        assert_eq!(
            dimensions.len(),
            2,
            "This proc block only supports 2D outputs (requested output: {:?})",
            dimensions
        );
    }
}
