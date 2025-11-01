// src/main.rs - rapih + debug out_vocab_dim
use anyhow::Result;
use tch::{nn, nn::OptimizerConfig, Device};

mod data_loader;
mod tokenizer;
mod model;
mod attention;
mod training;
mod metrics;

use data_loader::DataLoader;
use model::AttentionRNN;
use training::Trainer;
use metrics::{bleu_score, rouge_1}; // sentence_bleu dihapus

// Hyperparameters
const EMBEDDING_DIM: i64 = 256;
const HIDDEN_DIM: i64 = 512;
const NUM_LAYERS: i64 = 2;
const BATCH_SIZE: i64 = 8;
const MAX_SRC_LEN: i64 = 64;
const MAX_TGT_LEN: i64 = 64;
const EPOCHS: i64 = 10;
const LEARNING_RATE: f64 = 0.0001;

fn main() -> Result<()> {
    println!("========================================");
    println!("  Machine Translation with Attention RNN");
    println!("  Task: English → Indonesian (or any pair)");
    println!("========================================\n");

    let device = if tch::Cuda::is_available() {
        println!("✓ CUDA device detected");
        Device::Cuda(0)
    } else {
        println!("✓ Using CPU");
        Device::Cpu
    };
    println!("✓ CPU cores: {}\n", num_cpus::get());

    // Load data (beberapa path)
    println!("Loading translation data...");
    let csv_paths = [
        "translation_data.csv",
        "../Translate500.csv",
    ];

    let mut data_loader = None;
    for path in &csv_paths {
        println!("  Trying: {}", path);
        match DataLoader::new(path, MAX_SRC_LEN as usize, MAX_TGT_LEN as usize) {
            Ok(loader) => {
                data_loader = Some(loader);
                println!("  ✓ Successfully loaded from {}", path);
                break;
            }
            Err(e) => {
                println!("    ✗ Failed: {}", e);
            }
        }
    }

    let data_loader = data_loader.expect(
        "\n Could not find any valid CSV file!\n\
         Please create one of:\n  \
         - translation_data.csv (in project root)\n  \
         - Translate500.csv (one level up)\n\n\
         Format: CSV with 2 columns (source, target)"
    );

    println!("✓ Loaded {} sentence pairs", data_loader.num_samples());
    println!("✓ Unique source sentences: {}", data_loader.source_unique_count());
    println!("✓ Unique target sentences: {}\n", data_loader.target_unique_count());

    // Dual tokenizer
    println!("Building dual tokenizer (source + target)...");
    let mut dual_tokenizer = tokenizer::DualTokenizer::new(10_000, 10_000);
    dual_tokenizer.fit(&data_loader)?;

    let src_vocab_size = dual_tokenizer.source_vocab_size() as i64;
    let tgt_vocab_size = dual_tokenizer.target_vocab_size() as i64;

    println!("✓ Source vocabulary size: {}", src_vocab_size);
    println!("✓ Target vocabulary size: {}\n", tgt_vocab_size);

    // Split data
    let (train_data, val_data) = data_loader.train_val_split(0.9)?;
    println!("✓ Training pairs: {}", train_data.len());
    println!("✓ Validation pairs: {}\n", val_data.len());

    // Model
    println!("Initializing Attention RNN with dual vocabularies...");
    let mut vs = nn::VarStore::new(device);
    vs.set_kind(tch::Kind::Float);

    let model = AttentionRNN::new(
        &vs.root(),
        src_vocab_size,
        tgt_vocab_size, // PENTING: vocab TARGET, bukan MAX_TGT_LEN
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        0.3,
    );

    // Debug out_vocab_dim
    println!("✓ Model out_vocab_dim (should match target): {}\n", model.out_vocab_dim());

    // Hitung parameter
    let total_params: i64 = vs.trainable_variables().iter().map(|t| t.size().iter().product::<i64>()).sum();
    println!("✓ Model initialized with {} trainable parameters\n", total_params);

    // Optimizer
    let opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;

    // Trainer
    let mut trainer = Trainer::new(model, vs, opt, device, BATCH_SIZE as usize);

    // Train
    println!("Starting training...");
    println!("{}\n", "=".repeat(50));

    trainer.train(
        &train_data,
        &val_data,
        &dual_tokenizer,
        EPOCHS as usize,
        MAX_SRC_LEN as usize,
        MAX_TGT_LEN as usize,
    )?;

    // Save
    println!("\nSaving model...");
    trainer.save_model("translation_model.pt")?;
    println!("✓ Model saved to translation_model.pt");

    // Test
    println!("\n{}", "=".repeat(50));
    println!("Testing translation on validation samples...");
    println!("{}\n", "=".repeat(50));
    test_translation(&mut trainer, &val_data, &dual_tokenizer, device)?;

    Ok(())
}

fn test_translation(
    trainer: &mut Trainer,
    val_data: &[(String, String)],
    dual_tokenizer: &tokenizer::DualTokenizer,
    device: Device,
) -> Result<()> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let samples: Vec<_> = val_data.choose_multiple(&mut rng, 5).collect();

    let mut total_bleu = 0.0;
    let mut total_rouge = 0.0;

    for (i, (source, reference)) in samples.iter().enumerate() {
        println!("\n--- Sample {} ---", i + 1);
        println!("Source:    {}", source);
        println!("Reference: {}", reference);

        let translation = trainer.generate_summary(
            source,
            dual_tokenizer,
            MAX_TGT_LEN as usize,
            device,
        )?;

        println!("Generated: {}", translation);

        let bleu = bleu_score(reference, &translation);
        println!(
            "\nBLEU Scores:\n  BLEU-1: {:.4}  BLEU-2: {:.4}  BLEU-3: {:.4}  BLEU-4: {:.4}",
            bleu.bleu_1, bleu.bleu_2, bleu.bleu_3, bleu.bleu_4
        );
        println!("  Overall BLEU: {:.4}", bleu.bleu);

        let rouge = rouge_1(reference, &translation);
        println!(
            "  ROUGE-1: P={:.3} R={:.3} F1={:.3}",
            rouge.precision, rouge.recall, rouge.f1
        );
        println!("{}", "-".repeat(50));

        total_bleu += bleu.bleu;
        total_rouge += rouge.f1;
    }

    println!("\n{}", "=".repeat(50));
    println!("Average BLEU:    {:.4}", total_bleu / samples.len() as f64);
    println!("Average ROUGE-1: {:.4}", total_rouge / samples.len() as f64);
    println!("{}", "=".repeat(50));

    Ok(())
}
