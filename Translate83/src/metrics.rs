// src/metrics.rs
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct BleuScore {
    pub bleu_1: f64,
    pub bleu_2: f64,
    pub bleu_3: f64,
    pub bleu_4: f64,
    pub bleu: f64, // Geometric mean of 1-4 grams
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(|t| t.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|t| !t.is_empty())
        .collect()
}

fn ngrams(tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
    let mut map = HashMap::new();
    if n == 0 || tokens.len() < n {
        return map;
    }
    for i in 0..=tokens.len() - n {
        let key = tokens[i..i + n].to_vec();
        *map.entry(key).or_insert(0) += 1;
    }
    map
}

/// Calculate modified n-gram precision (BLEU component)
fn modified_precision(reference: &[String], candidate: &[String], n: usize) -> f64 {
    let ref_ngrams = ngrams(reference, n);
    let cand_ngrams = ngrams(candidate, n);

    if cand_ngrams.is_empty() {
        return 0.0;
    }

    let mut clipped_count = 0;
    let mut total_count = 0;

    for (ng, &cnt_cand) in cand_ngrams.iter() {
        total_count += cnt_cand;
        let cnt_ref = ref_ngrams.get(ng).copied().unwrap_or(0);
        clipped_count += cnt_cand.min(cnt_ref);
    }

    if total_count == 0 {
        0.0
    } else {
        clipped_count as f64 / total_count as f64
    }
}

/// Calculate BLEU score with brevity penalty
pub fn bleu_score(reference: &str, candidate: &str) -> BleuScore {
    let ref_tok = tokenize(reference);
    let cand_tok = tokenize(candidate);

    if ref_tok.is_empty() || cand_tok.is_empty() {
        return BleuScore::default();
    }

    // Calculate precision for n=1,2,3,4
    let p1 = modified_precision(&ref_tok, &cand_tok, 1);
    let p2 = modified_precision(&ref_tok, &cand_tok, 2);
    let p3 = modified_precision(&ref_tok, &cand_tok, 3);
    let p4 = modified_precision(&ref_tok, &cand_tok, 4);

    // Brevity penalty
    let c = cand_tok.len() as f64;
    let r = ref_tok.len() as f64;
    let bp = if c > r {
        1.0
    } else if c > 0.0 {
        (1.0 - r / c).exp()
    } else {
        0.0
    };

    // Geometric mean of precisions (skip if any is zero)
    let bleu = if p1 > 0.0 && p2 > 0.0 && p3 > 0.0 && p4 > 0.0 {
        bp * (p1 * p2 * p3 * p4).powf(0.25)
    } else {
        0.0
    };

    BleuScore {
        bleu_1: p1,
        bleu_2: p2,
        bleu_3: p3,
        bleu_4: p4,
        bleu,
    }
}

/// Calculate sentence-level BLEU (smoothed for short sentences)
pub fn sentence_bleu(reference: &str, candidate: &str) -> f64 {
    let score = bleu_score(reference, candidate);
    score.bleu
}

// ===== ROUGE metrics (optional, for comparison) =====
#[derive(Clone, Copy, Debug, Default)]
pub struct Rouge {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

fn rouge_ngrams(reference: &str, candidate: &str, n: usize) -> Rouge {
    let ref_tok = tokenize(reference);
    let cand_tok = tokenize(candidate);

    let ref_ng = ngrams(&ref_tok, n);
    let cand_ng = ngrams(&cand_tok, n);

    let ref_total: usize = ref_ng.values().sum();
    let cand_total: usize = cand_ng.values().sum();

    if ref_total == 0 || cand_total == 0 {
        return Rouge::default();
    }

    let mut overlap = 0;
    for (ng, &cnt_ref) in ref_ng.iter() {
        if let Some(&cnt_c) = cand_ng.get(ng) {
            overlap += cnt_ref.min(cnt_c);
        }
    }

    let precision = overlap as f64 / cand_total as f64;
    let recall = overlap as f64 / ref_total as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    Rouge { precision, recall, f1 }
}

pub fn rouge_1(reference: &str, candidate: &str) -> Rouge {
    rouge_ngrams(reference, candidate, 1)
}