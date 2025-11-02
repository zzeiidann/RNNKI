// src/model.rs
// This file is deprecated. Use model_transformer.rs instead.
// Kept for backward compatibility reference only.

use tch::{nn, nn::Module, nn::RNN, nn::LSTMState, Tensor, Kind};
use crate::attention::MultiHeadAttention;

pub struct AttentionRNN {
    embedding: nn::Embedding,
    encoder_lstm: nn::LSTM,
    decoder_lstm: nn::LSTM,
    attention: MultiHeadAttention,
    decoder_hidden_projection: nn::Linear,
    output_projection: nn::Linear,
    dropout: f64,
    hidden_dim: i64,
    num_layers: i64,
}

impl AttentionRNN {
    pub fn new(
        vs: &nn::Path,
        vocab_size: i64,
        embedding_dim: i64,
        hidden_dim: i64,
        num_layers: i64,
        dropout: f64,
    ) -> Self {
        Self::new_with_heads(vs, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, 8)
    }
    
    /// Constructor dengan parameter num_heads
    pub fn new_with_heads(
        vs: &nn::Path,
        vocab_size: i64,
        embedding_dim: i64,
        hidden_dim: i64,
        num_layers: i64,
        dropout: f64,
        num_heads: i64,
    ) -> Self {
        let embedding = nn::embedding(
            vs / "embedding",
            vocab_size,
            embedding_dim,
            Default::default(),
        );

        // Encoder: bi-LSTM (default batch_first=false, input [T,B,E])
        let mut encoder_config = nn::RNNConfig::default();
        encoder_config.dropout = dropout;
        encoder_config.num_layers = num_layers;
        encoder_config.bidirectional = true;

        let encoder_lstm = nn::lstm(
            vs / "encoder",
            embedding_dim,
            hidden_dim,
            encoder_config,
        );

        // Decoder: uni-LSTM, input concat(emb, context [2H])
        let mut decoder_config = nn::RNNConfig::default();
        decoder_config.dropout = dropout;
        decoder_config.num_layers = num_layers;

        let decoder_lstm = nn::lstm(
            vs / "decoder",
            embedding_dim + hidden_dim * 2,
            hidden_dim,
            decoder_config,
        );

        // Multi-Head Attention di atas keluaran bi-encoder (2H)
        let attention = MultiHeadAttention::new(
            &(vs / "attention"),
            hidden_dim * 2,
            num_heads,
            dropout,
        );

        // Proyeksi hidden decoder (H) -> dim encoder (2H)
        let decoder_hidden_projection = nn::linear(
            vs / "decoder_hidden_proj",
            hidden_dim,
            hidden_dim * 2,
            Default::default(),
        );

        // Proyeksi ke vocab
        let output_projection = nn::linear(
            vs / "output",
            hidden_dim,
            vocab_size,
            Default::default(),
        );

        Self {
            embedding,
            encoder_lstm,
            decoder_lstm,
            attention,
            decoder_hidden_projection,
            output_projection,
            dropout,
            hidden_dim,
            num_layers,
        }
    }

    /// Encode input sequence
    /// encoder_input: [B, T]
    /// returns: encoder_outputs [B, T, 2H], decoder_init_state
    pub fn encode(&self, input: &Tensor) -> (Tensor, LSTMState) {
        let embedded = self.embedding.forward(input);
        let embedded_t = embedded.transpose(0, 1);

        let (encoder_outputs_tbh, _enc_state) = self.encoder_lstm.seq(&embedded_t);
        let encoder_outputs = encoder_outputs_tbh.transpose(0, 1).contiguous();

        let b = encoder_outputs.size()[0];
        let device = input.device();
        let dec_h0 = Tensor::zeros(&[self.num_layers, b, self.hidden_dim], (Kind::Float, device));
        let dec_c0 = Tensor::zeros(&[self.num_layers, b, self.hidden_dim], (Kind::Float, device));
        let dec_state = LSTMState((dec_h0, dec_c0));

        (encoder_outputs, dec_state)
    }

    /// Decode one step with multi-head attention
    /// input: [B, 1]
    /// state: LSTMState
    /// encoder_outputs: [B, T, 2H]
    /// mask: Option<[B, T]>
    pub fn decode_step(
        &self,
        input: &Tensor,
        state: &LSTMState,
        encoder_outputs: &Tensor,
        mask: Option<&Tensor>,
    ) -> (Tensor, LSTMState, Tensor) {
        let embedded = self.embedding.forward(input); // [B, 1, E]

        let (hidden, _cell) = &state.0;
        let query = hidden.select(0, self.num_layers - 1); // [B, H]

        // Proyeksikan query ke dim 2H
        let query_proj = self.decoder_hidden_projection.forward(&query); // [B, 2H]

        // Multi-head attention
        // keys dan values sama-sama menggunakan encoder_outputs
        let (context, attention_weights) = self.attention.forward(
            &query_proj,
            encoder_outputs,
            encoder_outputs,
            mask,
        );

        let context_expanded = context.unsqueeze(1); // [B, 1, 2H]

        // Safety check sebelum concatenation
        let emb_sz = embedded.size();
        let ctx_sz = context_expanded.size();
        if emb_sz[0] != ctx_sz[0] || emb_sz[1] != ctx_sz[1] {
            eprintln!(
                "\n[ATTN SHAPE MISMATCH] embedded={:?} context={:?}\n",
                emb_sz, ctx_sz
            );
            panic!("Shape mismatch before concatenation");
        }

        let decoder_input = Tensor::cat(&[embedded, context_expanded], 2); // [B, 1, E+2H]
        let decoder_input_t = decoder_input.transpose(0, 1); // [1, B, E+2H]

        let (output_tbh, new_state) = self.decoder_lstm.seq_init(&decoder_input_t, state);
        let output = output_tbh.transpose(0, 1).squeeze_dim(1); // [B, H]

        let logits = self.output_projection.forward(&output); // [B, V]

        (logits, new_state, attention_weights)
    }

    /// Forward pass untuk training (teacher forcing)
    pub fn forward(
        &self,
        encoder_input_raw: &Tensor,
        decoder_input_raw: &Tensor,
        encoder_mask_raw: Option<&Tensor>,
    ) -> Tensor {
        // Normalisasi encoder_input ke [B, Ts]
        let encoder_input = if encoder_input_raw.dim() == 2
            && encoder_input_raw.size()[0] > encoder_input_raw.size()[1]
        {
            encoder_input_raw.transpose(0, 1).contiguous()
        } else {
            encoder_input_raw.shallow_clone()
        };

        // Normalisasi encoder_mask
        let encoder_mask = encoder_mask_raw.map(|m| {
            if m.dim() == 2 {
                if m.size()[0] == encoder_input.size()[0] && m.size()[1] == encoder_input.size()[1] {
                    m.shallow_clone()
                } else if m.size()[0] == encoder_input.size()[1] && m.size()[1] == encoder_input.size()[0] {
                    m.transpose(0, 1).contiguous()
                } else {
                    m.view([encoder_input.size()[0], encoder_input.size()[1]])
                }
            } else {
                m.view([encoder_input.size()[0], encoder_input.size()[1]])
            }
        });

        let batch_size = encoder_input.size()[0];

        // Normalisasi decoder_input ke [B, Tt]
        let decoder_input = if decoder_input_raw.dim() == 2 {
            if decoder_input_raw.size()[0] == batch_size {
                decoder_input_raw.shallow_clone()
            } else if decoder_input_raw.size()[1] == batch_size {
                decoder_input_raw.transpose(0, 1).contiguous()
            } else {
                decoder_input_raw.shallow_clone()
            }
        } else {
            let t_total = (decoder_input_raw.numel() as i64) / batch_size;
            decoder_input_raw.view([batch_size, t_total])
        };

        let tgt_seq_len = decoder_input.size()[1];

        // Encode
        let (encoder_outputs, mut state) = self.encode(&encoder_input);

        // Decode dengan teacher forcing
        let mut outputs = Vec::with_capacity(tgt_seq_len as usize);

        for t in 0..tgt_seq_len {
            let input_t = decoder_input.narrow(1, t, 1);
            let (logits, new_state, _attn) =
                self.decode_step(&input_t, &state, &encoder_outputs, encoder_mask.as_ref());
            outputs.push(logits.unsqueeze(1));
            state = new_state;
        }

        Tensor::cat(&outputs, 1) // [B, Tt, V]
    }
}