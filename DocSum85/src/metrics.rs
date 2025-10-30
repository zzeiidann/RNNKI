// src/metrics.rs
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rouge {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

fn tokenize(s: &str) -> Vec<String> {
    // lowercase + split whitespace + buang non-alnum (simple & cepat)
    s.to_lowercase()
        .split_whitespace()
        .map(|t| t.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|t| !t.is_empty())
        .collect()
}

fn ngrams(tokens: &[String], n: usize) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    if n == 0 || tokens.len() < n {
        return map;
    }
    for i in 0..=tokens.len() - n {
        let key = tokens[i..i + n].join(" ");
        *map.entry(key).or_insert(0) += 1;
    }
    map
}

pub fn rouge_n(reference: &str, candidate: &str, n: usize) -> Rouge {
    let ref_tok = tokenize(reference);
    let cand_tok = tokenize(candidate);

    let ref_ngrams = ngrams(&ref_tok, n);
    let cand_ngrams = ngrams(&cand_tok, n);

    let ref_total: usize = ref_ngrams.values().sum();
    let cand_total: usize = cand_ngrams.values().sum();

    if ref_total == 0 || cand_total == 0 {
        return Rouge::default();
    }

    let mut overlap: usize = 0;
    for (ng, &cnt_ref) in ref_ngrams.iter() {
        if let Some(&cnt_c) = cand_ngrams.get(ng) {
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

fn lcs_len(a: &[String], b: &[String]) -> usize {
    // DP dua-baris, O(|a|*|b|), cukup untuk ringkasan pendek
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return 0;
    }
    let mut prev = vec![0usize; m + 1];
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n {
        for j in 1..=m {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

pub fn rouge_l(reference: &str, candidate: &str) -> Rouge {
    let ref_tok = tokenize(reference);
    let cand_tok = tokenize(candidate);

    if ref_tok.is_empty() || cand_tok.is_empty() {
        return Rouge::default();
    }

    let lcs = lcs_len(&ref_tok, &cand_tok) as f64;
    let prec = lcs / (cand_tok.len() as f64);
    let rec = lcs / (ref_tok.len() as f64);
    let f1 = if prec + rec > 0.0 {
        2.0 * prec * rec / (prec + rec)
    } else {
        0.0
    };
    Rouge { precision: prec, recall: rec, f1 }
}

pub fn rouge_all(reference: &str, candidate: &str) -> (Rouge, Rouge, Rouge) {
    (rouge_n(reference, candidate, 1),
     rouge_n(reference, candidate, 2),
     rouge_l(reference, candidate))
}

pub fn rouge_1_f1(reference: &str, candidate: &str) -> f64 {
    rouge_n(reference, candidate, 1).f1
}
