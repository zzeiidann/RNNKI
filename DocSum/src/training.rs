// src/training.rs
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tch::{nn, Device, Kind, Tensor};

use crate::model::AttentionRNN;
use crate::tokenizer::Tokenizer;

pub struct Trainer {
    model: AttentionRNN,
    vs: nn::VarStore,
    opt: nn::Optimizer,
    device: Device,
    batch_size: usize,
}

impl Trainer {
    pub fn new(
        model: AttentionRNN,
        vs: nn::VarStore,
        opt: nn::Optimizer,
        device: Device,
        batch_size: usize,
    ) -> Self {
        Self { model, vs, opt, device, batch_size }
    }

    /// Encode batch + bikin mask
    fn prepare_batch(
        &self,
        batch: &[(String, String)],
        tokenizer: &Tokenizer,
        max_text_len: usize,
        max_summary_len: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        // Encode paralel
        let encoded: Vec<_> = batch
            .par_iter()
            .map(|(text, summary)| {
                // Asumsi: PAD=0
                let src        = tokenizer.encode_with_special_tokens(text, max_text_len,  false, true);
                let tgt_input  = tokenizer.encode_with_special_tokens(summary, max_summary_len, true,  false);
                let tgt_output = tokenizer.encode_with_special_tokens(summary, max_summary_len, false, true);
                // Mask sumber: 1(valid), 0(pad)
                let src_mask: Vec<i64> = src.iter().map(|&idx| if idx == 0 { 0 } else { 1 }).collect();
                (src, tgt_input, tgt_output, src_mask)
            })
            .collect();

        let b  = batch.len() as i64;
        let sl = max_text_len as i64;
        let tl = max_summary_len as i64;

        let src_data: Vec<i64>     = encoded.iter().flat_map(|(s,   _,   _, _)| s.iter().copied().map(|x| x as i64)).collect();
        let tgt_in_data: Vec<i64>  = encoded.iter().flat_map(|(_,   t_i, _, _)| t_i.iter().copied().map(|x| x as i64)).collect();
        let tgt_out_data: Vec<i64> = encoded.iter().flat_map(|(_,   _,   t_o, _)| t_o.iter().copied().map(|x| x as i64)).collect();
        let mask_data: Vec<i64>    = encoded.iter().flat_map(|(_,   _,   _, m)| m.iter().copied()).collect();

        let src        = Tensor::from_slice(&src_data).view([b, sl]).to(self.device);
        let tgt_input  = Tensor::from_slice(&tgt_in_data).view([b, tl]).to(self.device);
        let tgt_output = Tensor::from_slice(&tgt_out_data).view([b, tl]).to(self.device);
        // Mask boolean (true=valid, false=pad)
        let mask       = Tensor::from_slice(&mask_data).view([b, sl]).to_kind(Kind::Bool).to(self.device);

        (src, tgt_input, tgt_output, mask)
    }

    /// Masked CE (tanpa gather/ masked_select / broadcast).
    /// - Ignore PAD=0
    /// - Ignore OOV (target >= V)
    fn masked_cross_entropy_logits(
        &self,
        logits_flat: &Tensor,  // [N, V]
        targets_flat: &Tensor, // [N] (int64)
        pad_id: i64,
    ) -> Tensor {
        let device = logits_flat.device();
        let n = logits_flat.size()[0];
        let v = logits_flat.size()[1];

        // target long di device yg sama
        let tgt = targets_flat.to_device(device).to_kind(Kind::Int64); // [N]

        // valid: bukan PAD & di dalam vocab
        let valid_mask = tgt.ne(pad_id).logical_and(&tgt.lt(v)); // [N] Bool
        // ambil index baris valid (M bisa 0)
        let idx = valid_mask.nonzero(); // [M,1] atau [] kalau kosong

        if idx.numel() == 0 {
            return Tensor::zeros(&[], (Kind::Float, device));
        }

        // squeeze ke [M]
        let idx = idx.squeeze_dim(-1);

        // subset baris valid
        let logits_sel  = logits_flat.index_select(0, &idx); // [M, V]
        let targets_sel = tgt.index_select(0, &idx);         // [M]

        // log-softmax
        let log_probs = logits_sel.log_softmax(-1, Kind::Float); // [M, V]

        // linear index (baris [0..M))
        let m = log_probs.size()[0];
        let rows = Tensor::arange(m, (Kind::Int64, device));     // [M]
        let lin_idx = &rows * v + &targets_sel;                  // [M]

        // pick log p(y)
        let picked = log_probs.flatten(0, 1).index_select(0, &lin_idx); // [M]
        let nll = -picked;                                              // [M]

        nll.mean(Kind::Float)
    }

    /// Satu epoch training
    fn train_epoch(
        &mut self,
        data: &[(String, String)],
        tokenizer: &Tokenizer,
        max_text_len: usize,
        max_summary_len: usize,
    ) -> Result<f64> {
        self.vs.freeze();
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
        let mut num_batches_processed = 0usize;
        let pad_id: i64 = 0;

        for batch in data.chunks(self.batch_size) {
            let (src, tgt_input, tgt_output, mask) =
                self.prepare_batch(batch, tokenizer, max_text_len, max_summary_len);

            // Forward
            let logits = self.model.forward(&src, &tgt_input, Some(&mask)); // [B,T,V]
            debug_assert_eq!(logits.size().len(), 3, "logits must be [B,T,V]");
            debug_assert_eq!(tgt_output.size().len(), 2, "targets must be [B,T]");

            // Flatten aman
            let logits_flat  = logits.flatten(0, 1);     // [B*T, V]
            let targets_flat = tgt_output.flatten(0, 1); // [B*T]

            let loss = self.masked_cross_entropy_logits(&logits_flat, &targets_flat, pad_id);

            self.opt.zero_grad();
            loss.backward();
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

    /// Validasi
    fn validate(
        &mut self,
        data: &[(String, String)],
        tokenizer: &Tokenizer,
        max_text_len: usize,
        max_summary_len: usize,
    ) -> Result<f64> {
        self.vs.freeze();

        let mut total_loss = 0.0;
        let mut num_batches = 0usize;
        let pad_id: i64 = 0;

        tch::no_grad(|| {
            for batch in data.chunks(self.batch_size) {
                let (src, tgt_input, tgt_output, mask) =
                    self.prepare_batch(batch, tokenizer, max_text_len, max_summary_len);

                let logits = self.model.forward(&src, &tgt_input, Some(&mask)); // [B,T,V]

                let logits_flat  = logits.flatten(0, 1);     // [B*T, V]
                let targets_flat = tgt_output.flatten(0, 1);  // [B*T]

                let loss = self.masked_cross_entropy_logits(&logits_flat, &targets_flat, pad_id);
                total_loss += f64::try_from(&loss).unwrap_or(0.0);
                num_batches += 1;
            }
        });

        Ok(total_loss / num_batches as f64)
    }

    /// Loop training utama
    pub fn train(
        &mut self,
        train_data: &[(String, String)],
        val_data: &[(String, String)],
        tokenizer: &Tokenizer,
        epochs: usize,
        max_text_len: usize,
        max_summary_len: usize,
    ) -> Result<()> {
        let mut best_val_loss = f64::INFINITY;

        for epoch in 1..=epochs {
            println!("Epoch {}/{}", epoch, epochs);
            println!("{}", "-".repeat(50));

            let train_loss = self.train_epoch(train_data, tokenizer, max_text_len, max_summary_len)?;
            let val_loss   = self.validate(val_data, tokenizer, max_text_len, max_summary_len)?;

            println!("Train Loss: {:.4} | Val Loss: {:.4}", train_loss, val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                self.vs.save("best_model.pt")?;
                println!("✓ Saved best model (val_loss: {:.4})", val_loss);
            }
            println!();
        }

        self.vs.load("best_model.pt")?;
        println!("✓ Loaded best model");
        Ok(())
    }

    /// Greedy decode untuk generate ringkasan
    pub fn generate_summary(
        &mut self,
        text: &str,
        tokenizer: &Tokenizer,
        max_len: usize,
        device: Device,
    ) -> Result<String> {
        self.vs.freeze();

        let result = tch::no_grad(|| {
            // Encode input
            let src_tokens = tokenizer.encode_with_special_tokens(text, 512, false, true);
            let src_mask: Vec<i64> = src_tokens.iter().map(|&idx| if idx == 0 { 0 } else { 1 }).collect();

            let src_tensor = Tensor::from_slice(
                &src_tokens.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            )
            .view([1, src_tokens.len() as i64])
            .to(device);
            let mask_tensor = Tensor::from_slice(&src_mask)
                .view([1, src_mask.len() as i64])
                .to_kind(Kind::Bool)
                .to(device);

            // Encode
            let (encoder_outputs, mut state) = self.model.encode(&src_tensor);

            // Start dengan SOS
            let mut current_token = tokenizer.sos_token_id() as i64;
            let mut generated_tokens = Vec::new();

            for _ in 0..max_len {
                let input = Tensor::from_slice(&[current_token]).view([1, 1]).to(device);
                let (logits, new_state, _) =
                    self.model.decode_step(&input, &state, &encoder_outputs, Some(&mask_tensor));
                state = new_state;

                let predicted = logits.argmax(-1, false);
                current_token = i64::try_from(&predicted).unwrap_or(tokenizer.eos_token_id() as i64);
                if current_token == tokenizer.eos_token_id() as i64 { break; }
                generated_tokens.push(current_token as usize);
            }

            tokenizer.decode(&generated_tokens)
        });

        Ok(result)
    }

    pub fn save_model(&self, path: &str) -> Result<()> { self.vs.save(path)?; Ok(()) }
    pub fn load_model(&mut self, path: &str) -> Result<()> { self.vs.load(path)?; Ok(()) }
}
