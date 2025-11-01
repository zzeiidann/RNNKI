// src/model.rs
use tch::{nn, nn::Module, Tensor, Kind};

/// Multi-Head Self-Attention untuk Transformer
pub struct MultiHeadAttention {
    q_linear: nn::Linear,
    k_linear: nn::Linear,
    v_linear: nn::Linear,
    out_linear: nn::Linear,
    n_heads: i64,
    d_k: i64,
    dropout_rate: f64,
}

impl MultiHeadAttention {
    pub fn new(vs: &nn::Path, d_model: i64, n_heads: i64, dropout: f64) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;

        Self {
            q_linear: nn::linear(vs / "q", d_model, d_model, Default::default()),
            k_linear: nn::linear(vs / "k", d_model, d_model, Default::default()),
            v_linear: nn::linear(vs / "v", d_model, d_model, Default::default()),
            out_linear: nn::linear(vs / "out", d_model, d_model, Default::default()),
            n_heads,
            d_k,
            dropout_rate: dropout,
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        let (batch_size, seq_len, _d_model) = (x.size()[0], x.size()[1], x.size()[2]);

        // Linear projections: [B, S, D] -> [B, S, D]
        let q = self.q_linear.forward(x);
        let k = self.k_linear.forward(x);
        let v = self.v_linear.forward(x);

        // Reshape for multi-head: [B, S, D] -> [B, H, S, D_k]
        let q = q.view([batch_size, seq_len, self.n_heads, self.d_k])
            .transpose(1, 2);
        let k = k.view([batch_size, seq_len, self.n_heads, self.d_k])
            .transpose(1, 2);
        let v = v.view([batch_size, seq_len, self.n_heads, self.d_k])
            .transpose(1, 2);

        // Scaled dot-product attention: Q @ K^T / sqrt(d_k)
        let scale = (self.d_k as f64).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)) / scale; // [B, H, S, S]

        // Apply causal mask (untuk autoregressive generation)
        let scores = if let Some(m) = mask {
            // mask shape: [S, S] -> broadcast ke [B, H, S, S]
            let mask_expanded = m.unsqueeze(0).unsqueeze(0);
            scores.masked_fill(&mask_expanded.eq(0), -1e9)
        } else {
            scores
        };

        // Softmax dan dropout
        let mut attn = scores.softmax(-1, Kind::Float);
        if train && self.dropout_rate > 0.0 {
            attn = attn.dropout(self.dropout_rate, train);
        }

        // Apply attention to values: [B, H, S, S] @ [B, H, S, D_k] -> [B, H, S, D_k]
        let output = attn.matmul(&v);

        // Concatenate heads: [B, H, S, D_k] -> [B, S, D]
        let output = output.transpose(1, 2)
            .contiguous()
            .view([batch_size, seq_len, self.n_heads * self.d_k]);

        self.out_linear.forward(&output)
    }
}

/// Feed-Forward Network
pub struct FeedForward {
    linear1: nn::Linear,
    linear2: nn::Linear,
    dropout_rate: f64,
}

impl FeedForward {
    pub fn new(vs: &nn::Path, d_model: i64, d_ff: i64, dropout: f64) -> Self {
        Self {
            linear1: nn::linear(vs / "fc1", d_model, d_ff, Default::default()),
            linear2: nn::linear(vs / "fc2", d_ff, d_model, Default::default()),
            dropout_rate: dropout,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let mut x = self.linear1.forward(x).gelu("none");
        if train && self.dropout_rate > 0.0 {
            x = x.dropout(self.dropout_rate, train);
        }
        self.linear2.forward(&x)
    }
}

/// Transformer Block
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    dropout_rate: f64,
}

impl TransformerBlock {
    pub fn new(vs: &nn::Path, d_model: i64, n_heads: i64, d_ff: i64, dropout: f64) -> Self {
        Self {
            attention: MultiHeadAttention::new(&(vs / "attn"), d_model, n_heads, dropout),
            feed_forward: FeedForward::new(&(vs / "ff"), d_model, d_ff, dropout),
            norm1: nn::layer_norm(vs / "norm1", vec![d_model], Default::default()),
            norm2: nn::layer_norm(vs / "norm2", vec![d_model], Default::default()),
            dropout_rate: dropout,
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        // Self-attention dengan residual connection (Pre-LN)
        let x_norm = self.norm1.forward(x);
        let attn_out = self.attention.forward(&x_norm, mask, train);
        let attn_out = if train && self.dropout_rate > 0.0 {
            attn_out.dropout(self.dropout_rate, train)
        } else {
            attn_out
        };
        let x = x + attn_out;

        // Feed-forward dengan residual connection
        let x_norm = self.norm2.forward(&x);
        let ff_out = self.feed_forward.forward(&x_norm, train);
        let ff_out = if train && self.dropout_rate > 0.0 {
            ff_out.dropout(self.dropout_rate, train)
        } else {
            ff_out
        };
        x + ff_out
    }
}

/// Transformer Model untuk Text Generation
pub struct TransformerModel {
    embedding: nn::Embedding,
    pos_embedding: nn::Embedding,
    blocks: Vec<TransformerBlock>,
    norm: nn::LayerNorm,
    output_projection: nn::Linear,
    d_model: i64,
    max_seq_len: i64,
}

impl TransformerModel {
    pub fn new(
        vs: &nn::Path,
        vocab_size: i64,
        d_model: i64,
        n_heads: i64,
        n_layers: i64,
        d_ff: i64,
        max_seq_len: i64,
        dropout: f64,
    ) -> Self {
        let embedding = nn::embedding(
            vs / "token_emb",
            vocab_size,
            d_model,
            Default::default(),
        );

        let pos_embedding = nn::embedding(
            vs / "pos_emb",
            max_seq_len,
            d_model,
            Default::default(),
        );

        let mut blocks = Vec::new();
        for i in 0..n_layers {
            blocks.push(TransformerBlock::new(
                &(vs / format!("block_{}", i)),
                d_model,
                n_heads,
                d_ff,
                dropout,
            ));
        }

        let norm = nn::layer_norm(vs / "norm", vec![d_model], Default::default());
        
        let output_projection = nn::linear(
            vs / "output",
            d_model,
            vocab_size,
            Default::default(),
        );

        Self {
            embedding,
            pos_embedding,
            blocks,
            norm,
            output_projection,
            d_model,
            max_seq_len,
        }
    }

    /// Create causal mask untuk autoregressive generation
    fn create_causal_mask(seq_len: i64, device: tch::Device) -> Tensor {
        let mask = Tensor::ones(&[seq_len, seq_len], (Kind::Float, device))
            .tril(0); // Lower triangular
        mask
    }

    pub fn forward(&self, input: &Tensor, train: bool) -> Tensor {
        let (batch_size, seq_len) = (input.size()[0], input.size()[1]);
        let device = input.device();

        // Token embeddings
        let tok_emb = self.embedding.forward(input); // [B, S, D]

        // Positional embeddings
        let positions = Tensor::arange(seq_len, (Kind::Int64, device))
            .unsqueeze(0)
            .expand(&[batch_size, seq_len], false);
        let pos_emb = self.pos_embedding.forward(&positions);

        // Combine embeddings
        let mut x = tok_emb + pos_emb;

        // Create causal mask
        let mask = Self::create_causal_mask(seq_len, device);

        // Pass through transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, Some(&mask), train);
        }

        // Final layer norm
        x = self.norm.forward(&x);

        // Project to vocabulary
        self.output_projection.forward(&x) // [B, S, V]
    }

    pub fn generate(
        &self,
        start_tokens: &Tensor,
        max_len: i64,
        temperature: f64,
        eos_token_id: i64,
    ) -> Tensor {
        let mut tokens = start_tokens.shallow_clone();

        for _ in 0..max_len {
            let seq_len = tokens.size()[1];
            if seq_len >= self.max_seq_len {
                break;
            }

            // Forward pass (no grad)
            let logits = tch::no_grad(|| self.forward(&tokens, false));
            
            // Get logits for last position
            let last_logits = logits.narrow(1, seq_len - 1, 1).squeeze_dim(1); // [B, V]

            // Apply temperature
            let scaled_logits = &last_logits / temperature;
            
            // Sample from distribution
            let probs = scaled_logits.softmax(-1, Kind::Float);
            let next_token = probs.multinomial(1, false); // [B, 1]

            // Check for EOS before appending
            let next_token_val = i64::try_from(&next_token.shallow_clone()).unwrap_or(0);
            
            // Append to sequence
            tokens = Tensor::cat(&[tokens, next_token], 1);

            if next_token_val == eos_token_id {
                break;
            }
        }

        tokens
    }
}