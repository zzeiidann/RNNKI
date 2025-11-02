// src/main.rs
use anyhow::Result;
use tch::{nn, nn::OptimizerConfig, Device};

mod data_loader;
mod tokenizer;
mod model_transformer;
mod transformer;
mod attention;
mod training;
mod metrics;

use data_loader::DataLoader;
use tokenizer::Tokenizer;
use model_transformer::TransformerSeq2Seq;
use training::Trainer;
use metrics::{rouge_all, rouge_1_f1};

const EMBEDDING_DIM: i64 = 256;
const HIDDEN_DIM: i64 = 512;
const NUM_LAYERS: i64 = 2;
const NUM_HEADS: i64 = 8;       // Number of attention heads
const FF_DIM: i64 = 2048;        // Feed-forward dimension
const BATCH_SIZE: i64 = 8;
const MAX_SEQ_LEN: i64 = 512;
const MAX_SUMMARY_LEN: i64 = 128;
const EPOCHS: i64 = 50;
const LEARNING_RATE: f64 = 0.0001;
const VOCAB_SIZE: usize = 20000;

fn main() -> Result<()> {
    println!("Document Summarization with Multi-Head Attention RNN");
    println!("====================================================\n");

    // Detect device
    let device = if tch::Cuda::is_available() {
        println!("✓ CUDA device detected and will be used");
        Device::Cuda(0)
    } else {
        println!("✓ Using CPU (CUDA not available)");
        Device::Cpu
    };

    println!("✓ Number of CPU cores: {}\n", num_cpus::get());

    let data_loader = DataLoader::new("../500Docsum.csv", MAX_SEQ_LEN as usize, MAX_SUMMARY_LEN as usize)?;

    println!("✓ Loaded {} samples", data_loader.num_samples());
    println!("✓ Text column: {} unique values", data_loader.text_unique_count());
    println!("✓ Summary column: {} unique values\n", data_loader.summary_unique_count());

    // Build tokenizer
    println!("Building tokenizer...");
    let mut tokenizer = Tokenizer::new(VOCAB_SIZE);
    tokenizer.fit(&data_loader)?;
    let vocab_size = tokenizer.vocab_size() as i64;
    println!("✓ Vocabulary size: {}\n", vocab_size);

    // Split data
    let (train_data, val_data) = data_loader.train_val_split(0.9)?;
    println!("✓ Training samples: {}", train_data.len());
    println!("✓ Validation samples: {}\n", val_data.len());

    // Create Transformer-based model with multi-head attention
    println!("Initializing Transformer-Based Multi-Head Attention model...");
    println!("✓ Number of attention heads: {}", NUM_HEADS);
    println!("✓ Hidden dimension: {}", HIDDEN_DIM);
    println!("✓ Feed-forward dimension: {}", FF_DIM);
    println!("✓ Number of layers: {}", NUM_LAYERS);
    println!("✓ Head dimension: {}\n", HIDDEN_DIM / NUM_HEADS);
    
    let vs = nn::VarStore::new(device);
    let model = TransformerSeq2Seq::new(
        &vs.root(),
        vocab_size,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_HEADS,
        FF_DIM,
        0.3,  // dropout
        device,
    );
    println!("✓ Transformer model initialized on {:?}\n", device);

    // Create optimizer
    let opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;

    // Create trainer
    let mut trainer = Trainer::new(
        model,
        vs,
        opt,
        device,
        BATCH_SIZE as usize,
    );

    // Training loop
    println!("Starting training...");
    println!("====================\n");

    trainer.train(
        &train_data,
        &val_data,
        &tokenizer,
        EPOCHS as usize,
        MAX_SEQ_LEN as usize,
        MAX_SUMMARY_LEN as usize,
    )?;

    // Save model
    println!("\nSaving model...");
    trainer.save_model("transformer_multihead_attention_model.pt")?;
    println!("✓ Model saved to transformer_multihead_attention_model.pt");

    // Test summarization
    println!("\nTesting summarization on validation samples...");
    test_summarization(&mut trainer, &val_data, &tokenizer, device)?;

    Ok(())
}

fn test_summarization(
    trainer: &mut Trainer,
    val_data: &[(String, String)],
    tokenizer: &Tokenizer,
    device: Device,
) -> Result<()> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let samples: Vec<_> = val_data.choose_multiple(&mut rng, 3).collect();

    for (i, (text, reference_summary)) in samples.iter().enumerate() {
        println!("\n--- Sample {} ---", i + 1);
        let preview = &text[..text.len().min(200)];
        println!("Original text: {}{}", preview, if text.len() > 200 { "..." } else { "" });
        println!("\nReference summary: {}", reference_summary);

        let predicted_summary = trainer.generate_summary(
            text,
            tokenizer,
            MAX_SUMMARY_LEN as usize,
            device,
        )?;

        println!("Generated summary: {}", predicted_summary);

        // Hitung ROUGE metrics
        let (r1, r2, rl) = rouge_all(reference_summary, &predicted_summary);
        let r1f = rouge_1_f1(reference_summary, &predicted_summary);
        println!(
            "ROUGE-1  P:{:.3} R:{:.3} F1:{:.3}   | ROUGE-2  P:{:.3} R:{:.3} F1:{:.3}   | ROUGE-L  P:{:.3} R:{:.3} F1:{:.3}",
            r1.precision, r1.recall, r1.f1,
            r2.precision, r2.recall, r2.f1,
            rl.precision, rl.recall, rl.f1
        );
        println!("ROUGE-1 F1 only: {:.3}", r1f);
        println!("---");
    }

    Ok(())
}