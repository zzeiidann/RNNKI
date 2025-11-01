// src/training.rs - robust logits handling + 2D reshape before CE
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tch::{nn, Device, Kind, Tensor};

use crate::model::AttentionRNN;
use crate::tokenizer::DualTokenizer;

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
        mut vs: nn::VarStore,
        opt: nn::Optimizer,
        device: Device,
        batch_size: usize,
    ) -> Self {
        vs.unfreeze();
        Self { model, vs, opt, device, batch_size }
    }

    fn prepare_batch(
        &self,
        batch: &[(String, String)],
        dual_tokenizer: &DualTokenizer,
        max_src_len: usize,
        max_tgt_len: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let src_pad = dual_tokenizer.source.pad_token_id() as usize;

        let encoded: Vec<_> = batch
            .par_iter()
            .map(|(src_text, tgt_text)| {
                let src = dual_tokenizer
                    .source
                    .encode_with_special_tokens(src_text, max_src_len, false, true);

                let tgt_input = dual_tokenizer
                    .target
                    .encode_with_special_tokens(tgt_text, max_tgt_len, true, false);
                let tgt_output = dual_tokenizer
                    .target
                    .encode_with_special_tokens(tgt_text, max_tgt_len, false, true);

                let src_mask: Vec<i64> = src.iter().map(|&idx| if idx == src_pad { 0 } else { 1 }).collect();

                (src, tgt_input, tgt_output, src_mask)
            })
            .collect();

        let b = batch.len() as i64;
        let sl = max_src_len as i64;
        let tl = max_tgt_len as i64;

        let src_data: Vec<i64> = encoded.iter().flat_map(|(s, _, _, _)| s.iter().copied().map(|x| x as i64)).collect();
        let tgt_in_data: Vec<i64> = encoded.iter().flat_map(|(_, t_i, _, _)| t_i.iter().copied().map(|x| x as i64)).collect();
        let tgt_out_data: Vec<i64> = encoded.iter().flat_map(|(_, _, t_o, _)| t_o.iter().copied().map(|x| x as i64)).collect();
        let mask_data: Vec<i64> = encoded.iter().flat_map(|(_, _, _, m)| m.iter().copied()).collect();

        let src = Tensor::from_slice(&src_data).view([b, sl]).to(self.device);
        let tgt_input = Tensor::from_slice(&tgt_in_data).view([b, tl]).to(self.device);
        let tgt_output = Tensor::from_slice(&tgt_out_data).view([b, tl]).to(self.device);
        let mask = Tensor::from_slice(&mask_data).view([b, sl]).to_kind(Kind::Bool).to(self.device);

        (src, tgt_input, tgt_output, mask)
    }

    /// CE untuk logits [N,V] vs target [N] (PAD dimask).
    /// Aman saat tidak ada token valid → kembalikan nol yang tetap attach ke graph.
    fn masked_cross_entropy_logits(
        &self,
        logits_flat: &Tensor, // [N,V]
        targets_flat: &Tensor, // [N]
        pad_id: i64,
    ) -> Tensor {
        let device = logits_flat.device();
        let v = *logits_flat.size().last().unwrap(); // V dari dimensi terakhir (robust)

        let tgt = targets_flat.to_device(device).to_kind(Kind::Int64);
        let valid_mask = tgt.ne(pad_id).logical_and(&tgt.lt(v));

        // Anchor nol yang tetap attach ke graph
        let zero_anchor = logits_flat.sum(Kind::Float) * 0.0;

        let idx = valid_mask.nonzero();
        if idx.numel() == 0 {
            return zero_anchor;
        }

        let idx = idx.squeeze_dim(-1);                // [M]
        let logits_sel = logits_flat.index_select(0, &idx); // [M,V]
        let targets_sel = tgt.index_select(0, &idx);        // [M]

        let ce = logits_sel.cross_entropy_for_logits(&targets_sel); // scalar
        ce + zero_anchor
    }

    fn train_epoch(
        &mut self,
        data: &[(String, String)],
        dual_tokenizer: &DualTokenizer,
        max_src_len: usize,
        max_tgt_len: usize,
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

        let mut total_loss = 0.0f64;
        let mut num_batches_processed = 0usize;

        let pad_id: i64 = dual_tokenizer.target.pad_token_id() as i64;
        let mut first_batch = true;

        for batch in data.chunks(self.batch_size) {
            let (src, tgt_input, tgt_output, mask) =
                self.prepare_batch(batch, dual_tokenizer, max_src_len, max_tgt_len);

            self.opt.zero_grad();

            let logits_btV = self.model.forward(&src, &tgt_input, Some(&mask)); // ekspektasi [B,T,V] (3D)

            if first_batch {
                let v_logits = *logits_btV.size().last().unwrap(); // **ambil dimensi terakhir**
                let v_tok = dual_tokenizer.target.vocab_size() as i64;
                if v_logits != v_tok {
                    anyhow::bail!(
                        "Vocab mismatch: model V={} vs tokenizer TARGET V={}. Pastikan layer output model memakai ukuran vocab TARGET.",
                        v_logits, v_tok
                    );
                }
                first_batch = false;
            }

            // **Selalu** reshape ke 2D [N,V] sebelum CE, di mana N = B*T (gabung batch & time)
            let v = *logits_btV.size().last().unwrap();
            let logits_flat = logits_btV.view([-1, v]);        // [B*T, V]
            let targets_flat = tgt_output.flatten(0, 1);       // [B*T]

            if cfg!(debug_assertions) {
                let tgt = targets_flat.to_kind(Kind::Int64);
                let valid_mask = tgt.ne(pad_id).logical_and(&tgt.lt(v));
                let n_valid: i64 = i64::try_from(&valid_mask.sum(Kind::Int64)).unwrap_or(0);
                if n_valid == 0 {
                    eprintln!("⚠️  Batch tanpa token valid (semua PAD/out-of-range). Cek pad_id & vocab.");
                }
            }

            let loss = self.masked_cross_entropy_logits(&logits_flat, &targets_flat, pad_id);

            self.opt.backward_step(&loss);

            let loss_val = f64::try_from(&loss).unwrap_or(0.0);
            total_loss += loss_val;
            num_batches_processed += 1;
            pb.set_message(format!("Loss: {:.4}", loss_val));
            pb.inc(1);
        }

        pb.finish_with_message("Epoch complete");
        Ok(if num_batches_processed > 0 { total_loss / num_batches_processed as f64 } else { 0.0 })
    }

    fn validate(
        &mut self,
        data: &[(String, String)],
        dual_tokenizer: &DualTokenizer,
        max_src_len: usize,
        max_tgt_len: usize,
    ) -> Result<f64> {
        let mut total_loss = 0.0f64;
        let mut num_batches = 0usize;

        let pad_id: i64 = dual_tokenizer.target.pad_token_id() as i64;

        tch::no_grad(|| {
            for batch in data.chunks(self.batch_size) {
                let (src, tgt_input, tgt_output, mask) =
                    self.prepare_batch(batch, dual_tokenizer, max_src_len, max_tgt_len);

                let logits_btV = self.model.forward(&src, &tgt_input, Some(&mask)); // [B,T,V]
                let v = *logits_btV.size().last().unwrap();
                let logits_flat = logits_btV.view([-1, v]);    // [B*T,V]
                let targets_flat = tgt_output.flatten(0, 1);   // [B*T]

                let loss = self.masked_cross_entropy_logits(&logits_flat, &targets_flat, pad_id);
                total_loss += f64::try_from(&loss).unwrap_or(0.0);
                num_batches += 1;
            }
        });

        Ok(if num_batches > 0 { total_loss / num_batches as f64 } else { 0.0 })
    }

    pub fn train(
        &mut self,
        train_data: &[(String, String)],
        val_data: &[(String, String)],
        dual_tokenizer: &DualTokenizer,
        epochs: usize,
        max_src_len: usize,
        max_tgt_len: usize,
    ) -> Result<()> {
        let mut best_val_loss = f64::INFINITY;

        println!("Checking gradient setup...");
        let trainable_count = self.vs.trainable_variables().len();
        println!("✓ Found {} trainable variables", trainable_count);
        if trainable_count == 0 {
            anyhow::bail!("No trainable variables found! Model initialization failed.");
        }

        for epoch in 1..=epochs {
            println!("Epoch {}/{}", epoch, epochs);
            println!("{}", "-".repeat(50));

            let train_loss = self.train_epoch(train_data, dual_tokenizer, max_src_len, max_tgt_len)?;
            let val_loss = self.validate(val_data, dual_tokenizer, max_src_len, max_tgt_len)?;

            println!("Train Loss: {:.4} | Val Loss: {:.4}", train_loss, val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                self.vs.save("best_model.pt")?;
                println!("✓ Saved best model (val_loss: {:.4})", val_loss);
            }
            println!();
        }

        println!("✓ Training complete. Best val loss: {:.4}", best_val_loss);
        Ok(())
    }

    /// Greedy decoding (TARGET tokenizer)
    pub fn generate_summary(
        &mut self,
        text: &str,
        dual_tokenizer: &DualTokenizer,
        max_len: usize,
        device: Device,
    ) -> Result<String> {
        let result = tch::no_grad(|| {
            // Encode SOURCE
            let src_tokens = dual_tokenizer.source.encode_with_special_tokens(text, 512, false, true);
            let src_pad = dual_tokenizer.source.pad_token_id() as usize;
            let src_mask: Vec<i64> = src_tokens.iter().map(|&idx| if idx == src_pad { 0 } else { 1 }).collect();

            let src_tensor = Tensor::from_slice(&src_tokens.iter().map(|&x| x as i64).collect::<Vec<_>>())
                .view([1, src_tokens.len() as i64])
                .to(device);

            let mask_tensor = Tensor::from_slice(&src_mask)
                .view([1, src_mask.len() as i64])
                .to_kind(Kind::Bool)
                .to(device);

            let (encoder_outputs, mut state) = self.model.encode(&src_tensor);

            // Decode TARGET
            let sos = dual_tokenizer.target.sos_token_id() as i64;
            let eos = dual_tokenizer.target.eos_token_id() as i64;

            let mut current_token = sos;
            let mut generated_tokens = Vec::new();

            for _ in 0..max_len {
                let input = Tensor::from_slice(&[current_token]).view([1, 1]).to(device);
                let (logits, new_state, _) =
                    self.model.decode_step(&input, &state, &encoder_outputs, Some(&mask_tensor));
                state = new_state;

                let predicted = logits.argmax(-1, false); // [1]
                current_token = i64::try_from(&predicted).unwrap_or(eos);

                if current_token == eos {
                    break;
                }
                generated_tokens.push(current_token as usize);
            }

            dual_tokenizer.target.decode(&generated_tokens)
        });

        Ok(result)
    }

    pub fn save_model(&self, path: &str) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }
}
