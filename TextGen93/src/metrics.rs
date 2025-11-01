// src/metrics.rs
use std::collections::{HashMap, HashSet};

/// Perplexity - metrik utama untuk language modeling
pub fn perplexity(loss: f64) -> f64 {
    loss.exp()
}

/* -------------------- BLEU (dengan BP + smoothing) -------------------- */

fn ngram_counts(tokens: &[&str], n: usize) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    if tokens.len() < n { return map; }
    for i in 0..=tokens.len() - n {
        let ng = tokens[i..i + n].join(" ");
        *map.entry(ng).or_insert(0) += 1;
    }
    map
}

/// BLEU klasik dengan brevity penalty dan smoothing epsilon.
/// Mengembalikan: (bleu, p1..pN, bp)
pub fn bleu_bp_smooth(reference: &str, candidate: &str, max_n: usize, eps: f64) -> (f64, Vec<f64>, f64) {
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
    let cand_tokens: Vec<&str> = candidate.split_whitespace().collect();

    if cand_tokens.is_empty() || ref_tokens.is_empty() {
        return (0.0, vec![0.0; max_n], 0.0);
    }

    let mut precisions = Vec::with_capacity(max_n);
    for n in 1..=max_n {
        let ref_ng = ngram_counts(&ref_tokens, n);
        let cand_ng = ngram_counts(&cand_tokens, n);

        let cand_total: usize = cand_ng.values().sum();
        if cand_total == 0 {
            precisions.push(0.0);
            continue;
        }

        let mut matched = 0usize;
        for (ng, &cnt) in cand_ng.iter() {
            if let Some(rc) = ref_ng.get(ng) {
                matched += cnt.min(*rc);
            }
        }
        let p_n = (matched as f64 + eps) / (cand_total as f64 + eps);
        precisions.push(p_n);
    }

    // Brevity penalty
    let c = cand_tokens.len() as f64;
    let r = ref_tokens.len() as f64;
    let bp = if c < r { f64::exp(1.0 - r / c) } else { 1.0 };

    // Geometric mean
    let geo = if precisions.iter().any(|&p| p <= 0.0) {
        0.0
    } else {
        f64::exp(precisions.iter().map(|p| p.ln()).sum::<f64>() / (max_n as f64))
    };

    (bp * geo, precisions, bp)
}

/// Convenience: BLEU-n dengan BP (geometric mean sampai n, sesuai definisi BLEU)
pub fn bleu_n_with_bp(reference: &str, candidate: &str, n: usize) -> f64 {
    let (bleu, _, _) = bleu_bp_smooth(reference, candidate, n, 1e-9);
    bleu
}

/* -------------------- (Legacy) clipped n-gram precision tanpa BP -------------------- */
/* Dibiarkan ada jika masih dipakai tempat lain. Tidak dipakai lagi di GenerationMetrics. */
pub fn bleu_score(reference: &str, candidate: &str, n: usize) -> f64 {
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
    let cand_tokens: Vec<&str> = candidate.split_whitespace().collect();

    if cand_tokens.is_empty() || ref_tokens.len() < n {
        return 0.0;
    }

    let ref_ngrams = get_ngrams(&ref_tokens, n);
    let cand_ngrams = get_ngrams(&cand_tokens, n);

    let mut matches = 0;
    for (ng, count) in cand_ngrams.iter() {
        if let Some(&ref_count) = ref_ngrams.get(ng) {
            matches += (*count).min(ref_count);
        }
    }

    let total_cand_ngrams: usize = cand_ngrams.values().sum();
    if total_cand_ngrams == 0 { 0.0 } else { matches as f64 / total_cand_ngrams as f64 }
}

fn get_ngrams(tokens: &[&str], n: usize) -> HashMap<String, usize> {
    let mut ngrams = HashMap::new();
    if tokens.len() < n { return ngrams; }
    for i in 0..=tokens.len() - n {
        let ngram = tokens[i..i + n].join(" ");
        *ngrams.entry(ngram).or_insert(0) += 1;
    }
    ngrams
}

/* -------------------- Diversity & repetition -------------------- */

/// Diversity metrics - unique n-grams ratio (per-teks)
pub fn diversity_score(text: &str, n: usize) -> f64 {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < n { return 0.0; }

    let mut ngrams = HashSet::new();
    let mut total = 0usize;

    for i in 0..=tokens.len() - n {
        let ngram = tokens[i..i + n].join(" ");
        ngrams.insert(ngram);
        total += 1;
    }

    if total == 0 { 0.0 } else { ngrams.len() as f64 / total as f64 }
}

/// Self-BLEU: rata-rata BLEU antar output (lebih rendah = lebih beragam)
pub fn self_bleu(texts: &[String], n: usize) -> f64 {
    if texts.len() < 2 { return 0.0; }
    let mut total = 0.0;
    let mut cnt = 0usize;
    for i in 0..texts.len() {
        for j in 0..texts.len() {
            if i == j { continue; }
            // pakai BLEU dengan BP biar stabil
            total += bleu_n_with_bp(&texts[i], &texts[j], n);
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { total / cnt as f64 }
}

/// Coherence sederhana: repetisi trigram (lebih rendah = lebih koheren)
pub fn repetition_score(text: &str) -> f64 {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < 3 { return 0.0; }

    let mut counts = HashMap::<String, usize>::new();
    for i in 0..=tokens.len() - 3 {
        let g = format!("{} {} {}", tokens[i], tokens[i+1], tokens[i+2]);
        *counts.entry(g).or_insert(0) += 1;
    }
    let total = (tokens.len() - 2) as f64;
    let repeated: usize = counts.values().filter(|&&v| v > 1).sum();
    repeated as f64 / total.max(1.0)
}

/* -------------------- Bundle untuk laporan cepat -------------------- */

#[derive(Debug, Clone)]
pub struct GenerationMetrics {
    pub perplexity: f64,
    pub bleu_1: f64,      // sudah mengandung BP
    pub bleu_2: f64,      // sudah mengandung BP
    pub diversity_2: f64,
    pub diversity_3: f64,
    pub repetition: f64,  // trigram repetition rate
}

impl GenerationMetrics {
    pub fn calculate(
        loss: f64,
        reference: Option<&str>,
        candidate: &str,
    ) -> Self {
        let ppl = perplexity(loss);

        let (bleu_1, bleu_2) = if let Some(ref_text) = reference {
            (
                bleu_n_with_bp(ref_text, candidate, 1),
                bleu_n_with_bp(ref_text, candidate, 2),
            )
        } else {
            (0.0, 0.0)
        };

        Self {
            perplexity: ppl,
            bleu_1,
            bleu_2,
            diversity_2: diversity_score(candidate, 2),
            diversity_3: diversity_score(candidate, 3),
            repetition: repetition_score(candidate),
        }
    }

    pub fn print(&self) {
        println!("\n Generation Metrics:");
        println!("  Perplexity: {:.2}", self.perplexity);
        println!("  BLEU-1 (BP): {:.3}", self.bleu_1);
        println!("  BLEU-2 (BP): {:.3}", self.bleu_2);
        println!("  Diversity-2: {:.3}", self.diversity_2);
        println!("  Diversity-3: {:.3}", self.diversity_3);
        println!("  Repetition (tri-gram): {:.3}", self.repetition);
    }
}

/// Compare Transformer vs RNN
pub fn print_comparison(
    transformer_metrics: &GenerationMetrics,
    rnn_metrics: Option<&GenerationMetrics>,
) {
    println!("\n Model Comparison:");
    println!("{}", "=".repeat(60));
    println!("{:<20} {:<15} {:<15}", "Metric", "Transformer", "RNN");
    println!("{}", "-".repeat(60));

    if let Some(rnn) = rnn_metrics {
        println!("{:<20} {:<15.2} {:<15.2}",
                 "Perplexity", transformer_metrics.perplexity, rnn.perplexity);
        println!("{:<20} {:<15.3} {:<15.3}",
                 "BLEU-1 (BP)", transformer_metrics.bleu_1, rnn.bleu_1);
        println!("{:<20} {:<15.3} {:<15.3}",
                 "BLEU-2 (BP)", transformer_metrics.bleu_2, rnn.bleu_2);
        println!("{:<20} {:<15.3} {:<15.3}",
                 "Diversity-2", transformer_metrics.diversity_2, rnn.diversity_2);
        println!("{:<20} {:<15.3} {:<15.3}",
                 "Repetition", transformer_metrics.repetition, rnn.repetition);
    } else {
        println!("{:<20} {:<15.2} {:<15}", "Perplexity", transformer_metrics.perplexity, "N/A");
        println!("{:<20} {:<15.3} {:<15}", "BLEU-1 (BP)", transformer_metrics.bleu_1, "N/A");
        println!("{:<20} {:<15.3} {:<15}", "BLEU-2 (BP)", transformer_metrics.bleu_2, "N/A");
        println!("{:<20} {:<15.3} {:<15}", "Diversity-2", transformer_metrics.diversity_2, "N/A");
        println!("{:<20} {:<15.3} {:<15}", "Repetition", transformer_metrics.repetition, "N/A");
    }
    println!("{}", "=".repeat(60));
}
