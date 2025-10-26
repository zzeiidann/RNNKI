// src/data_loader.rs
use anyhow::{Result, Context};
use csv::ReaderBuilder;
use serde::Deserialize;
use std::collections::HashSet;
use rayon::prelude::*;

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "Text")]
    text: String,
    #[serde(rename = "Summary")]
    summary: String,
}

pub struct DataLoader {
    data: Vec<(String, String)>,
    max_text_len: usize,
    max_summary_len: usize,
    text_unique: usize,
    summary_unique: usize,
}

impl DataLoader {
    pub fn new(path: &str, max_text_len: usize, max_summary_len: usize) -> Result<Self> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .context("Failed to open CSV file")?;

        let records: Vec<Record> = reader
            .deserialize()
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to deserialize CSV records")?;

        // Process data in parallel
        let data: Vec<(String, String)> = records
            .par_iter()
            .map(|record| {
                let text = Self::clean_text(&record.text);
                let summary = Self::clean_text(&record.summary);
                (text, summary)
            })
            .filter(|(text, summary)| !text.is_empty() && !summary.is_empty())
            .collect();

        // Calculate unique values
        let text_unique = data.iter().map(|(t, _)| t).collect::<HashSet<_>>().len();
        let summary_unique = data.iter().map(|(_, s)| s).collect::<HashSet<_>>().len();

        Ok(Self {
            data,
            max_text_len,
            max_summary_len,
            text_unique,
            summary_unique,
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
        self.data.len()
    }

    pub fn text_unique_count(&self) -> usize {
        self.text_unique
    }

    pub fn summary_unique_count(&self) -> usize {
        self.summary_unique
    }

    pub fn get_all_data(&self) -> &[(String, String)] {
        &self.data
    }

    pub fn train_val_split(&self, train_ratio: f32) -> Result<(Vec<(String, String)>, Vec<(String, String)>)> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut data = self.data.clone();
        let mut rng = thread_rng();
        data.shuffle(&mut rng);

        let split_idx = (data.len() as f32 * train_ratio) as usize;
        let (train, val) = data.split_at(split_idx);

        Ok((train.to_vec(), val.to_vec()))
    }

    pub fn max_text_len(&self) -> usize {
        self.max_text_len
    }

    pub fn max_summary_len(&self) -> usize {
        self.max_summary_len
    }
}