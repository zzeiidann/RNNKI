// src/data_loader.rs
use anyhow::{Result, Context};
use std::fs::File;
use std::io::{BufRead, BufReader};
use rayon::prelude::*;

pub struct DataLoader {
    sequences: Vec<String>,
    max_seq_len: usize,
}

impl DataLoader {
    /// Load text file untuk text generation
    /// Format: satu baris = satu sequence/sentence
    pub fn new(path: &str, max_seq_len: usize) -> Result<Self> {
        let file = File::open(path)
            .context("Failed to open text file")?;
        
        let reader = BufReader::new(file);
        
        // Read dan clean setiap baris
        let sequences: Vec<String> = reader
            .lines()
            .filter_map(|line| line.ok())
            .par_bridge()
            .map(|line| Self::clean_text(&line))
            .filter(|text| !text.is_empty() && text.split_whitespace().count() >= 3)
            .collect();

        println!("✓ Loaded {} sequences from {}", sequences.len(), path);
        
        Ok(Self {
            sequences,
            max_seq_len,
        })
    }

    /// Load dari CSV dengan kolom "text"
    pub fn from_csv(path: &str, max_seq_len: usize, _text_column: &str) -> Result<Self> {
        use csv::ReaderBuilder;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .context("Failed to open CSV file")?;

        let sequences: Vec<String> = reader
            .records()
            .filter_map(|record| record.ok())
            .par_bridge()
            .filter_map(|record| {
                // Cari kolom text
                record.get(0).map(|s| Self::clean_text(s))
            })
            .filter(|text| !text.is_empty() && text.split_whitespace().count() >= 3)
            .collect();

        println!("✓ Loaded {} sequences from CSV", sequences.len());
        
        Ok(Self {
            sequences,
            max_seq_len,
        })
    }

    fn clean_text(text: &str) -> String {
        text.trim()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?;:'-".contains(*c))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    pub fn num_samples(&self) -> usize {
        self.sequences.len()
    }

    pub fn get_all_data(&self) -> &[String] {
        &self.sequences
    }

    pub fn train_val_split(&self, train_ratio: f32) -> Result<(Vec<String>, Vec<String>)> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut data = self.sequences.clone();
        let mut rng = thread_rng();
        data.shuffle(&mut rng);

        let split_idx = (data.len() as f32 * train_ratio) as usize;
        let (train, val) = data.split_at(split_idx);

        Ok((train.to_vec(), val.to_vec()))
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}