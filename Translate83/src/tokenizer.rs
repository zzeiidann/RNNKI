// src/tokenizer.rs - DUAL TOKENIZER VERSION
use anyhow::Result;
use std::collections::HashMap;
use rayon::prelude::*;
use crate::data_loader::DataLoader;

pub const PAD_TOKEN: &str = "<PAD>";
pub const UNK_TOKEN: &str = "<UNK>";
pub const SOS_TOKEN: &str = "<SOS>";
pub const EOS_TOKEN: &str = "<EOS>";

/// Single language tokenizer
pub struct Tokenizer {
    word2idx: HashMap<String, usize>,
    idx2word: HashMap<usize, String>,
    max_vocab_size: usize,
}

impl Tokenizer {
    pub fn new(max_vocab_size: usize) -> Self {
        let mut word2idx = HashMap::new();
        let mut idx2word = HashMap::new();

        // Special tokens
        word2idx.insert(PAD_TOKEN.to_string(), 0);
        word2idx.insert(UNK_TOKEN.to_string(), 1);
        word2idx.insert(SOS_TOKEN.to_string(), 2);
        word2idx.insert(EOS_TOKEN.to_string(), 3);

        idx2word.insert(0, PAD_TOKEN.to_string());
        idx2word.insert(1, UNK_TOKEN.to_string());
        idx2word.insert(2, SOS_TOKEN.to_string());
        idx2word.insert(3, EOS_TOKEN.to_string());

        Self { word2idx, idx2word, max_vocab_size }
    }

    /// Build vocabulary from text iterator
    pub fn fit_from_texts<'a, I>(&mut self, texts: I) -> Result<()>
    where
        I: Iterator<Item = &'a String> + Clone,
    {
        // Count word frequencies
        let word_counts: HashMap<String, usize> = texts
            .flat_map(|text| text.split_whitespace().map(String::from))
            .fold(HashMap::new(), |mut acc, word| {
                *acc.entry(word).or_insert(0) += 1;
                acc
            });

        // Sort by frequency
        let mut word_freq: Vec<_> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));

        // Add top words
        let max_words = self.max_vocab_size - 4;
        for (word, _) in word_freq.iter().take(max_words) {
            let idx = self.word2idx.len();
            self.word2idx.insert(word.clone(), idx);
            self.idx2word.insert(idx, word.clone());
        }

        Ok(())
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| *self.word2idx.get(word).unwrap_or(&1))
            .collect()
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter_map(|&idx| {
                let word = self.idx2word.get(&idx)?;
                if word == PAD_TOKEN || word == SOS_TOKEN || word == EOS_TOKEN {
                    None
                } else {
                    Some(word.as_str())
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_size(&self) -> usize {
        self.word2idx.len()
    }

    pub fn pad_token_id(&self) -> usize { 0 }
    pub fn sos_token_id(&self) -> usize { 2 }
    pub fn eos_token_id(&self) -> usize { 3 }

    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        max_len: usize,
        add_sos: bool,
        add_eos: bool,
    ) -> Vec<usize> {
        let mut tokens = if add_sos {
            vec![self.sos_token_id()]
        } else {
            Vec::new()
        };

        tokens.extend(self.encode(text));

        if add_eos && tokens.len() < max_len {
            tokens.push(self.eos_token_id());
        }

        tokens.truncate(max_len);

        while tokens.len() < max_len {
            tokens.push(self.pad_token_id());
        }

        tokens
    }
}

/// Dual tokenizer for machine translation (separate source/target vocabularies)
pub struct DualTokenizer {
    pub source: Tokenizer,
    pub target: Tokenizer,
}

impl DualTokenizer {
    pub fn new(source_vocab_size: usize, target_vocab_size: usize) -> Self {
        Self {
            source: Tokenizer::new(source_vocab_size),
            target: Tokenizer::new(target_vocab_size),
        }
    }

    /// Build vocabularies from parallel corpus
    pub fn fit(&mut self, data_loader: &DataLoader) -> Result<()> {
        let all_data = data_loader.get_all_data();

        println!("  Building source vocabulary...");
        let source_texts = all_data.iter().map(|(src, _)| src);
        self.source.fit_from_texts(source_texts)?;

        println!("  Building target vocabulary...");
        let target_texts = all_data.iter().map(|(_, tgt)| tgt);
        self.target.fit_from_texts(target_texts)?;

        Ok(())
    }

    pub fn source_vocab_size(&self) -> usize {
        self.source.vocab_size()
    }

    pub fn target_vocab_size(&self) -> usize {
        self.target.vocab_size()
    }
}