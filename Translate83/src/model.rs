// src/model.rs - DUAL VOCABULARY VERSION + robust decode_step
use tch::{nn, nn::Module, nn::LSTMState, nn::RNN, Kind, Tensor};
use crate::attention::BahdanauAttention;

pub struct AttentionRNN {
    // Embedding terpisah
    source_embedding: nn::Embedding,
    target_embedding: nn::Embedding,

    // Encoder bi-LSTM, decoder uni-LSTM
    encoder_lstm: nn::LSTM,
    decoder_lstm: nn::LSTM,

    // Attention dan proyeksi
    attention: BahdanauAttention,
    decoder_hidden_projection: nn::Linear, // H -> 2H (query untuk attention)
    output_projection: nn::Linear,         // H -> V (target vocab)

    // Hyper
    dropout: f64,
    hidden_dim: i64,
    num_layers: i64,
}

impl AttentionRNN {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vs: &nn::Path,
        source_vocab_size: i64,
        target_vocab_size: i64,
        embedding_dim: i64,
        hidden_dim: i64,
        num_layers: i64,
        dropout: f64,
    ) -> Self {
        println!(
            "[DEBUG:model::new] srcV={}, tgtV={}, emb={}, hid={}, layers={}, drop={}",
            source_vocab_size, target_vocab_size, embedding_dim, hidden_dim, num_layers, dropout
        );

        // Embeddings
        let source_embedding = nn::embedding(vs / "source_embedding", source_vocab_size, embedding_dim, Default::default());
        let target_embedding = nn::embedding(vs / "target_embedding", target_vocab_size, embedding_dim, Default::default());

        // Encoder: bi-LSTM (output 2H)
        let mut encoder_cfg = nn::RNNConfig::default();
        encoder_cfg.dropout = dropout;
        encoder_cfg.num_layers = num_layers;
        encoder_cfg.bidirectional = true;
        let encoder_lstm = nn::lstm(vs / "encoder", embedding_dim, hidden_dim, encoder_cfg);

        // Decoder: input = [emb, context(2H)] → dim = emb + 2H, hidden = H
        let mut decoder_cfg = nn::RNNConfig::default();
        decoder_cfg.dropout = dropout;
        decoder_cfg.num_layers = num_layers;
        let decoder_lstm = nn::lstm(vs / "decoder", embedding_dim + hidden_dim * 2, hidden_dim, decoder_cfg);

        // Attention di ruang 2H (encoder output)
        let attention = BahdanauAttention::new(&(vs / "attention"), hidden_dim * 2);

        // Proyeksi hidden decoder H → 2H (query attention)
        let decoder_hidden_projection = nn::linear(vs / "decoder_hidden_proj", hidden_dim, hidden_dim * 2, Default::default());

        // Output ke vocab target: H → V
        let output_projection = nn::linear(vs / "output", hidden_dim, target_vocab_size, Default::default());

        let out_v = output_projection.ws.size()[0];
        println!("[DEBUG:model::new] out_proj out_features={}", out_v);
        debug_assert_eq!(out_v, target_vocab_size, "out_proj features harus = target vocab");

        Self {
            source_embedding,
            target_embedding,
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

    /// Bantu debug: dimensi vocab output
    pub fn out_vocab_dim(&self) -> i64 {
        self.output_projection.ws.size()[0]
    }

    /// Encode dengan SOURCE embedding → (encoder_outputs [B,T,2H], init decoder state [layers,B,H])
    pub fn encode(&self, input: &Tensor) -> (Tensor, LSTMState) {
        // input: [B,T]
        let embedded = self.source_embedding.forward(input); // [B,T,E]
        let embedded_t = embedded.transpose(0, 1); // [T,B,E]

        let (encoder_outputs_tbh, _enc_state) = self.encoder_lstm.seq(&embedded_t); // [T,B,2H]
        let encoder_outputs = encoder_outputs_tbh.transpose(0, 1).contiguous();     // [B,T,2H]

        // Init decoder state nol
        let b = input.size()[0];
        let device = input.device();
        let dec_h0 = Tensor::zeros(&[self.num_layers, b, self.hidden_dim], (Kind::Float, device));
        let dec_c0 = Tensor::zeros(&[self.num_layers, b, self.hidden_dim], (Kind::Float, device));
        let dec_state = LSTMState((dec_h0, dec_c0));

        (encoder_outputs, dec_state)
    }

    /// Satu langkah decode dengan TARGET embedding + attention
    /// **Robust**: pastikan output 2D [B,H] lalu proyeksi → [B,V]
    pub fn decode_step(
        &self,
        input: &Tensor,                // [B,1]
        state: &LSTMState,             // (h,c) [layers,B,H]
        encoder_outputs: &Tensor,      // [B,T,2H]
        mask: Option<&Tensor>,         // [B,T] (1 valid, 0 pad)
    ) -> (Tensor, LSTMState, Tensor) {
        let b = encoder_outputs.size()[0];

        let embedded = self.target_embedding.forward(input); // [B,1,E]

        let (hidden, _cell) = &state.0;               // hidden: [layers,B,H]
        let query = hidden.select(0, self.num_layers - 1); // [B,H]
        let query_proj = self.decoder_hidden_projection.forward(&query); // [B,2H]

        let (context, attn_w) = self.attention.forward(&query_proj, encoder_outputs, mask); // [B,2H], [B,T]
        let context_expanded = context.unsqueeze(1); // [B,1,2H]

        // Input LSTM decoder: concat([emb, context]) → [B,1,E+2H]
        let decoder_input = Tensor::cat(&[embedded, context_expanded], 2); // [B,1,E+2H]
        let decoder_input_t = decoder_input.transpose(0, 1); // [1,B,E+2H] (RNN batch_first=false)

        let (out_any, new_state) = self.decoder_lstm.seq_init(&decoder_input_t, state);
        // out_any bisa [T,B,H] (= [1,B,H]) atau (jika batch_first true) [B,T,H] (= [B,1,H])
        // Kita buat **robust**: squeeze dimensi waktu yang = 1 dan pastikan hasil [B,H]
        let output = if out_any.dim() == 3 {
            let s = out_any.size();
            if s[0] == 1 && s[1] == b {
                // [1,B,H] → [B,H]
                out_any.squeeze_dim(0)
            } else if s[1] == 1 && s[0] == b {
                // [B,1,H] → [B,H]
                out_any.squeeze_dim(1)
            } else if s[0] == b {
                // [B,T,H] (T>1) → ambil step terakhir
                out_any.select(1, s[1] - 1)
            } else {
                // [T,B,H] (T>1) → ambil step terakhir
                out_any.select(0, s[0] - 1)
            }
        } else {
            // fallback (jarang terjadi)
            out_any
        }; // [B,H]

        let logits = self.output_projection.forward(&output); // [B,V]
        (logits, new_state, attn_w)
    }

    /// Forward penuh (teacher forcing): kembalikan logits [B,T,V]
    pub fn forward(
        &self,
        encoder_input_raw: &Tensor,      // [B,T]
        decoder_input_raw: &Tensor,      // [B,T]
        encoder_mask_raw: Option<&Tensor>,
    ) -> Tensor {
        // Normalisasi shape input encoder → [B,T]
        let encoder_input = if encoder_input_raw.dim() == 2 && encoder_input_raw.size()[0] > encoder_input_raw.size()[1] {
            encoder_input_raw.transpose(0, 1).contiguous()
        } else {
            encoder_input_raw.shallow_clone()
        };

        // Normalisasi mask → [B,T]
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

        // Normalisasi decoder input → [B,T]
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
        let (encoder_outputs, mut state) = self.encode(&encoder_input); // [B,T,2H], state: [layers,B,H]

        // Decode T langkah
        let mut outputs = Vec::with_capacity(tgt_seq_len as usize);
        for t in 0..tgt_seq_len {
            let input_t = decoder_input.narrow(1, t, 1); // [B,1]
            let (logits, new_state, _attn) = self.decode_step(&input_t, &state, &encoder_outputs, encoder_mask.as_ref());
            outputs.push(logits.unsqueeze(1)); // **selalu** [B,1,V]
            state = new_state;
        }

        Tensor::cat(&outputs, 1) // [B,T,V]
    }
}
