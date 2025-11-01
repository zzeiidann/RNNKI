// src/training.rs
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tch::{nn, Device, Kind, Tensor};

use crate::model::TransformerModel;
use crate::tokenizer::Tokenizer;

pub struct Trainer {
    model: TransformerModel,
    vs: nn::VarStore,
    opt: nn::Optimizer,
    device: Device,
    batch_size: usize,
}

impl Trainer {
    pub fn new(
        model: TransformerModel,
        vs: nn::VarStore,
        opt: nn::Optimizer,
        device: Device,
        batch_size: usize,
    ) -> Self {
        Self {
            model,
            vs,
            opt,
            device,
            batch_size,
        }
    }

    /// Prepare batch untuk autoregressive language modeling
    fn prepare_batch(
        &self,
        batch: &[String],
        tokenizer: &Tokenizer,
        max_seq_len: usize,
    ) -> (Tensor, Tensor) {
        // Encode sequences paralel
        let encoded: Vec<Vec<usize>> = batch
            .par_iter()
            .map(|text| {
                tokenizer.encode_with_special_tokens(text, max_seq_len, true, true)
            })
            .collect();

        let batch_size = batch.len() as i64;
        let seq_len = max_seq_len as i64;

        // Input: semua token kecuali yang terakhir
        let input_data: Vec<i64> = encoded
            .iter()
            .flat_map(|tokens| tokens[..tokens.len()-1].iter().map(|&t| t as i64))
            .collect();

        // Target: semua token kecuali yang pertama (shifted)
        let target_data: Vec<i64> = encoded
            .iter()
            .flat_map(|tokens| tokens[1..].iter().map(|&t| t as i64))
            .collect();

        let input = Tensor::from_slice(&input_data)
            .view([batch_size, seq_len - 1])
            .to(self.device);
        
        let target = Tensor::from_slice(&target_data)
            .view([batch_size, seq_len - 1])
            .to(self.device);

        (input, target)
    }

    /// Masked cross-entropy loss (ignore padding)
    fn masked_cross_entropy(
        &self,
        logits: &Tensor,   // [B, S, V]
        targets: &Tensor,  // [B, S]
        pad_id: i64,
    ) -> Tensor {
        let device = logits.device();
        let (batch_size, seq_len, vocab_size) = (
            logits.size()[0],
            logits.size()[1],
            logits.size()[2],
        );

        // Flatten: [B, S, V] -> [B*S, V], [B, S] -> [B*S]
        let logits_flat = logits.view([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.view([batch_size * seq_len])
            .to_device(device)
            .to_kind(Kind::Int64);

        // Create valid mask (non-padding tokens)
        let valid_mask = targets_flat.ne(pad_id).logical_and(&targets_flat.lt(vocab_size));
        let idx = valid_mask.nonzero();

        if idx.numel() == 0 {
            return Tensor::zeros(&[], (Kind::Float, device));
        }

        let idx = idx.squeeze_dim(-1);

        // Select valid logits and targets
        let logits_sel = logits_flat.index_select(0, &idx);
        let targets_sel = targets_flat.index_select(0, &idx);

        // Cross-entropy loss
        logits_sel.cross_entropy_for_logits(&targets_sel)
    }

    /// Train one epoch
    fn train_epoch(
        &mut self,
        data: &[String],
        tokenizer: &Tokenizer,
        max_seq_len: usize,
    ) -> Result<f64> {
        self.vs.unfreeze();

        let num_batches = (data.len() + self.batch_size - 1) / self.batch_size;
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        let mut total_loss = 0.0;
        let mut num_batches_processed = 0;
        let pad_id: i64 = 0;

        for batch in data.chunks(self.batch_size) {
            let (input, target) = self.prepare_batch(batch, tokenizer, max_seq_len);

            // Forward pass (training mode)
            let logits = self.model.forward(&input, true); // [B, S, V]
            
            // Compute loss
            let loss = self.masked_cross_entropy(&logits, &target, pad_id);

            // Backward pass
            self.opt.zero_grad();
            loss.backward();
            
            // Gradient clipping (manual implementation)
            for (_name, tensor) in self.vs.variables() {
                let mut grad = tensor.grad();
                if grad.defined() {
                    let _ = grad.clamp_(-1.0, 1.0);
                }
            }
            
            self.opt.step();

            let loss_val = f64::try_from(&loss).unwrap_or(0.0);
            total_loss += loss_val;
            num_batches_processed += 1;
            
            pb.set_message(format!("Loss: {:.4}", loss_val));
            pb.inc(1);
        }

        pb.finish_with_message("Epoch complete");
        Ok(total_loss / num_batches_processed as f64)
    }

    /// Validate
    fn validate(
        &mut self,
        data: &[String],
        tokenizer: &Tokenizer,
        max_seq_len: usize,
    ) -> Result<f64> {
        self.vs.freeze();

        let mut total_loss = 0.0;
        let mut num_batches = 0;
        let pad_id: i64 = 0;

        tch::no_grad(|| {
            for batch in data.chunks(self.batch_size) {
                let (input, target) = self.prepare_batch(batch, tokenizer, max_seq_len);

                // Forward pass (eval mode)
                let logits = self.model.forward(&input, false);
                
                // Compute loss
                let loss = self.masked_cross_entropy(&logits, &target, pad_id);
                
                total_loss += f64::try_from(&loss).unwrap_or(0.0);
                num_batches += 1;
            }
        });

        Ok(total_loss / num_batches as f64)
    }

    /// Main training loop
    pub fn train(
        &mut self,
        train_data: &[String],
        val_data: &[String],
        tokenizer: &Tokenizer,
        epochs: usize,
        max_seq_len: usize,
    ) -> Result<()> {
        let mut best_val_loss = f64::INFINITY;

        for epoch in 1..=epochs {
            println!("\nEpoch {}/{}", epoch, epochs);
            println!("{}", "=".repeat(50));

            let train_loss = self.train_epoch(train_data, tokenizer, max_seq_len)?;
            let val_loss = self.validate(val_data, tokenizer, max_seq_len)?;

            // Calculate perplexity
            let train_ppl = train_loss.exp();
            let val_ppl = val_loss.exp();

            println!(
                "Train Loss: {:.4} (PPL: {:.2}) | Val Loss: {:.4} (PPL: {:.2})",
                train_loss, train_ppl, val_loss, val_ppl
            );

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                self.vs.save("best_transformer_model.pt")?;
                println!("✓ Saved best model (val_loss: {:.4})", val_loss);
            }
        }

        println!("\n✓ Training complete! Best val loss: {:.4}", best_val_loss);
        Ok(())
    }

    /// Generate text
    pub fn generate_text(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        max_len: usize,
        temperature: f64,
    ) -> Result<String> {
        self.vs.freeze();

        let result = tch::no_grad(|| {
            // Encode prompt
            let tokens = tokenizer.encode_with_special_tokens(prompt, 64, true, false);
            let input = Tensor::from_slice(
                &tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
            )
            .view([1, tokens.len() as i64])
            .to(self.device);

            // Generate
            let generated = self.model.generate(
                &input,
                max_len as i64,
                temperature,
                tokenizer.eos_token_id() as i64,
            );

            // Decode
            let gen_vec: Vec<i64> = generated.view([-1]).try_into().unwrap_or_default();
            let gen_tokens: Vec<usize> = gen_vec.iter().map(|&x| x as usize).collect();
            
            tokenizer.decode(&gen_tokens)
        });

        Ok(result)
    }

    pub fn save_model(&self, path: &str) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    pub fn load_model(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }
}