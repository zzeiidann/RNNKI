// src/data_loader.rs
use anyhow::{Result, Context};
use csv::ReaderBuilder;
use std::collections::HashSet;

/// DataLoader untuk Machine Translation (EN → ID atau pasangan bahasa lain)
/// Expected CSV format: | source | target |
pub struct DataLoader {
    data: Vec<(String, String)>, // (source_sentence, target_sentence)
    max_src_len: usize,
    max_tgt_len: usize,
}

impl DataLoader {
    pub fn new(csv_path: &str, max_src_len: usize, max_tgt_len: usize) -> Result<Self> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(csv_path)
            .context("Failed to open CSV file")?;

        let mut data = Vec::new();

        for result in reader.records() {
            let record = result.context("Failed to read CSV record")?;
            
            // Expect columns: source, target
            if record.len() < 2 {
                continue; // Skip malformed rows
            }

            let source = record[0].trim().to_string();
            let target = record[1].trim().to_string();

            // Basic filtering: skip empty or too long sentences
            let src_words = source.split_whitespace().count();
            let tgt_words = target.split_whitespace().count();

            if source.is_empty() || target.is_empty() 
                || src_words > max_src_len || tgt_words > max_tgt_len {
                continue;
            }

            data.push((source, target));
        }

        Ok(Self {
            data,
            max_src_len,
            max_tgt_len,
        })
    }

    pub fn num_samples(&self) -> usize {
        self.data.len()
    }

    pub fn get_all_data(&self) -> &[(String, String)] {
        &self.data
    }

    /// Count unique source sentences
    pub fn source_unique_count(&self) -> usize {
        self.data.iter().map(|(s, _)| s).collect::<HashSet<_>>().len()
    }

    /// Count unique target sentences
    pub fn target_unique_count(&self) -> usize {
        self.data.iter().map(|(_, t)| t).collect::<HashSet<_>>().len()
    }

    /// Split data into train and validation sets
    pub fn train_val_split(&self, train_ratio: f64) -> Result<(Vec<(String, String)>, Vec<(String, String)>)> {
        let split_idx = (self.data.len() as f64 * train_ratio) as usize;
        
        let train = self.data[..split_idx].to_vec();
        let val = self.data[split_idx..].to_vec();

        Ok((train, val))
    }
}