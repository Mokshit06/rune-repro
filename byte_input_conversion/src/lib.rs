// #![no_std]
// extern crate alloc;

// #[cfg(test)]
// #[macro_use]
// extern crate std;

// #[cfg(not(test))]
// extern crate no_std_compat as std;

// #[cfg(not(test))]
// use std::prelude::v1::*;

// use alloc::string::String;
// pub use hotg_rune_core::{HasOutputs, Tensor};
// use hotg_rune_proc_blocks::{ProcBlock, Transform};
// use log::{debug, info};
// // rust-tokenizers -> serde-json-core
// // fs2
// use tokenizers::Tokenizer;

// #[derive(Debug, Clone, PartialEq, ProcBlock)]
// #[transform(input = [u8; _], output = [u8; _])]
// pub struct ByteInputConversion {}

// impl ByteInputConversion {
//     pub const fn new() -> Self {
//         ByteInputConversion {}
//     }
// }

// i want to build a 'url shortener'
// -> url hashes

// i want to make a 'website'

// i want to build a 'website' with 'auth'
// -> react, cookie, express, auth0, mongoose

// impl Default for ByteInputConversion {
//     fn default() -> Self {
//         ByteInputConversion::new()
//     }
// }

// impl Transform<Tensor<u8>> for ByteInputConversion {
//     type Output = Tensor<u8>;

//     fn transform(&mut self, input: Tensor<u8>) -> Self::Output {
//         let mut sentence = String::new();
//         let res = input.map(|_dims, value| {
//             // remove trailing null chars
//             if *value != 0 {
//                 sentence.push(*value as char);
//             };

//             *value
//         });

//         info!("{}", sentence);

//         // let vocab_path = "/root/projects/hack/htn/bert/tokenizer/vocab.txt";

//         let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
//         let encoding = tokenizer.encode(sentence, false).unwrap();
//         info!("Albert: {:?}", encoding.get_tokens());

//         // let vocab = BertVocab::from_file(&vocab_path).unwrap();

//         // let test_sentence = Example::new_from_string("This is a sample sentence to be tokenized");
//         // let bert_tokenizer: BertTokenizer = BertTokenizer::from_existing_vocab(vocab, true, true);

//         // let tokens = bert_tokenizer.encode(
//         //     &test_sentence.sentence_1,
//         //     None,
//         //     128,
//         //     &TruncationStrategy::LongestFirst,
//         //     0,
//         // );

//         // info!("{:?}", tokens);

//         res
//     }
// }

// impl HasOutputs for ByteInputConversion {
//     fn set_output_dimensions(&mut self, dimensions: &[usize]) {
//         assert_eq!(
//             dimensions.len(),
//             1,
//             "This proc block only supports 1D outputs (requested output: {:?})",
//             dimensions
//         );
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use alloc::vec;

//     #[test]
//     fn handle_empty() {
//         let mut pb = ByteInputConversion::new();
//         let input = Tensor::new_vector(vec![0; 15]);
//         let got = pb.transform(input);
//         assert_eq!(got.dimensions(), &[15]);
//     }

//     // #[test]
//     // fn does_it_match() {
//     //     let max = u8::MAX;
//     //     let min = u8::MIN;

//     //     let mut pb = ByteInputConversion::new();
//     //     let input = Tensor::new_vector(vec![0, max / 2, min / 2]);

//     //     let got = pb.transform(input);

//     //     assert_eq!(got.elements()[0..3], [0.0, 0.49998474, -0.50001526]);
//     // }

//     // #[test]
//     // fn does_clutch_work() {
//     //     let max = u8::MAX;
//     //     let min = u8::MIN;

//     //     let mut pb = ByteInputConversion::new();
//     //     let input = Tensor::new_vector(vec![max, min, min + 1]);

//     //     let got = pb.transform(input);

//     //     // assert_eq!(got.elements()[0..3], [1, -1, -1]);
//     //     assert_eq!(got.elements()[0..3], [1, 0, 0]);
//     // }
// }

// -----------------------------------------------------------------

#![no_std]

extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

pub use hotg_rune_core::{HasOutputs, Tensor};
use hotg_rune_proc_blocks::{ProcBlock, Transform};

// TODO: Add Generics

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
        input.map(|_dims, &value| value as i32)
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

// -----------------------------------------------------------------

// #![no_std]

// extern crate alloc;

// #[cfg(test)]
// #[macro_use]
// extern crate std;

// pub use hotg_rune_core::{HasOutputs, Tensor};
// use hotg_rune_proc_blocks::{ProcBlock, Transform};

// // TODO: Add Generics

// #[derive(Debug, Clone, PartialEq, ProcBlock)]
// #[transform(input = [u8; _], output = ([i32; 384], [i32; 384], [i32; 384]))]
// pub struct ByteInputConversion {}

// impl ByteInputConversion {
//     pub const fn new() -> Self {
//         ByteInputConversion {}
//     }
// }

// impl Default for ByteInputConversion {
//     fn default() -> Self {
//         ByteInputConversion::new()
//     }
// }

// impl Transform<Tensor<u8>> for ByteInputConversion {
//     type Output = (Tensor<i32>, Tensor<i32>, Tensor<i32>);

//     fn transform(&mut self, input: Tensor<u8>) -> Self::Output {
//         let val = input.map(|_dims, &value| value as i32);

//         (val.clone(), val.clone(), val.clone())
//     }
// }

// impl HasOutputs for ByteInputConversion {
//     fn set_output_dimensions(&mut self, dimensions: &[usize]) {
//         assert_eq!(
//             dimensions.len(),
//             2,
//             "This proc block only supports 2D outputs (requested output: {:?})",
//             dimensions
//         );
//     }
// }
