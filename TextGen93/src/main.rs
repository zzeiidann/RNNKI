// src/main.rs
use anyhow::Result;
use tch::{nn, nn::OptimizerConfig, Device};

mod data_loader;
mod tokenizer;
mod model;
mod training;
mod metrics;

use data_loader::DataLoader;
use tokenizer::Tokenizer;
use model::TransformerModel;
use training::Trainer;
use metrics::GenerationMetrics;

// ============= HYPERPARAMETERS =============
const D_MODEL: i64 = 256;          // Embedding dimension
const N_HEADS: i64 = 8;            // Number of attention heads
const N_LAYERS: i64 = 4;           // Number of transformer layers
const D_FF: i64 = 1024;            // Feed-forward dimension
const DROPOUT: f64 = 0.1;          // Dropout rate

const BATCH_SIZE: i64 = 32;
const MAX_SEQ_LEN: i64 = 128;
const EPOCHS: i64 = 1;
const LEARNING_RATE: f64 = 0.0001;
const VOCAB_SIZE: usize = 10000;

fn main() -> Result<()> {
    println!(" Transformer Model for Text Generation");
    println!("{}", "=".repeat(60));

    // Detect device
    let device = if tch::Cuda::is_available() {
        println!("✓ CUDA device detected");
        Device::Cuda(0)
    } else {
        println!("✓ Using CPU");
        Device::Cpu
    };

    // ============= LOAD DATA =============
    println!("\n Loading dataset...");
    
    let data_loader = DataLoader::new("../TextGen1.txt", MAX_SEQ_LEN as usize)?;

    println!("✓ Loaded {} sequences", data_loader.num_samples());

    // ============= BUILD TOKENIZER =============
    println!("\n Building tokenizer...");
    let mut tokenizer = Tokenizer::new(VOCAB_SIZE);
    tokenizer.fit(&data_loader)?;
    let vocab_size = tokenizer.vocab_size() as i64;
    println!("✓ Vocabulary size: {}", vocab_size);

    // ============= SPLIT DATA =============
    let (train_data, val_data) = data_loader.train_val_split(0.9)?;
    println!("\n Dataset split:");
    println!("  Training: {} sequences", train_data.len());
    println!("  Validation: {} sequences", val_data.len());

    // ============= CREATE TRANSFORMER MODEL =============
    println!("\n  Initializing Transformer model...");
    println!("  d_model: {}", D_MODEL);
    println!("  n_heads: {}", N_HEADS);
    println!("  n_layers: {}", N_LAYERS);
    println!("  d_ff: {}", D_FF);
    
    let vs = nn::VarStore::new(device);
    let model = TransformerModel::new(
        &vs.root(),
        vocab_size,
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        D_FF,
        MAX_SEQ_LEN,
        DROPOUT,
    );

    println!("✓ Model initialized on {:?}", device);
    
    // Count parameters
    let total_params: i64 = vs.variables().iter()
        .map(|(_, tensor)| tensor.size().iter().product::<i64>())
        .sum();
    println!("✓ Total parameters: {:.2}M", total_params as f64 / 1_000_000.0);

    // ============= CREATE OPTIMIZER =============
    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;
    opt.set_weight_decay(0.01); // Weight decay untuk regularization

    // ============= CREATE TRAINER =============
    let mut trainer = Trainer::new(
        model,
        vs,
        opt,
        device,
        BATCH_SIZE as usize,
    );

    // ============= TRAINING =============
    println!("\n Starting training...");
    println!("{}", "=".repeat(60));

    trainer.train(
        &train_data,
        &val_data,
        &tokenizer,
        EPOCHS as usize,
        MAX_SEQ_LEN as usize,
    )?;

    // ============= SAVE MODEL =============
    println!("\n Saving final model...");
    trainer.save_model("transformer_final.pt")?;
    println!("✓ Model saved to transformer_final.pt");

    // ============= TEXT GENERATION DEMO =============
    println!("\n Testing text generation...");
    println!("{}", "=".repeat(60));

    test_generation(&mut trainer, &val_data, &tokenizer)?;

    Ok(())
}

fn test_generation(
    trainer: &mut Trainer,
    val_data: &[String],
    tokenizer: &Tokenizer,
) -> Result<()> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    // Test dengan berbagai prompts
    let test_prompts = vec![
        "once upon a time",
        "the quick brown",
        "in the beginning",
    ];

    println!("\n Generation Examples:");
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n{}. Prompt: \"{}\"", i + 1, prompt);
        
        // Generate dengan temperature berbeda
        for temp in [0.7, 1.0, 1.2] {
            let generated = trainer.generate_text(
                prompt,
                tokenizer,
                50,  // max_len
                temp,
            )?;
            
            println!("   [T={:.1}] {}", temp, generated);
        }
    }

    // Test pada validation sample
    println!("\n Validation Samples:");
    let samples: Vec<_> = val_data.choose_multiple(&mut rng, 3).collect();
    
    let mut generated_texts = Vec::new();
    
    for (i, text) in samples.iter().enumerate() {
        println!("\n{}. Original:", i + 1);
        let preview = &text[..text.len().min(100)];
        println!("   {}{}", preview, if text.len() > 100 { "..." } else { "" });

        // Ambil beberapa kata pertama sebagai prompt
        let words: Vec<&str> = text.split_whitespace().take(3).collect();
        let prompt = words.join(" ");

        let generated = trainer.generate_text(
            &prompt,
            tokenizer,
            60,
            0.8,  // temperature
        )?;

        println!("   Generated:");
        println!("   {}", generated);

        // Calculate metrics
        let metrics = GenerationMetrics::calculate(0.0, Some(text), &generated);
        println!("\n   Metrics:");
        println!("     BLEU-1: {:.3}", metrics.bleu_1);
        println!("     BLEU-2: {:.3}", metrics.bleu_2);
        println!("     Diversity-2: {:.3}", metrics.diversity_2);
        println!("     Repetition: {:.3}", metrics.repetition);

        generated_texts.push(generated);
    }

    // Self-BLEU for diversity
    let self_bleu = metrics::self_bleu(&generated_texts, 2);
    println!("\n Overall Diversity (Self-BLEU-2): {:.3}", self_bleu);
    println!("   (Lower is more diverse)");

    Ok(())
}