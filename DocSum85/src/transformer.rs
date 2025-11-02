// src/transformer.rs
use tch::{nn, nn::Module, Tensor, Kind};

/// Positional Encoding
pub struct PositionalEncoding {
    pe: Tensor,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: i64, hidden_dim: i64, device: tch::Device) -> Self {
        let mut pe = vec![vec![0.0f32; hidden_dim as usize]; max_seq_len as usize];
        
        for pos in 0..max_seq_len {
            for i in (0..hidden_dim as usize).step_by(2) {
                let pos_f = pos as f64;
                let i_f = i as f64;
                let div_term = (10000.0f64).powf(i_f / hidden_dim as f64);
                
                pe[pos as usize][i] = (pos_f / div_term).sin() as f32;
                if i + 1 < hidden_dim as usize {
                    pe[pos as usize][(i + 1) as usize] = (pos_f / div_term).cos() as f32;
                }
            }
        }
        
        let pe_flat: Vec<f32> = pe.iter().flat_map(|row| row.clone()).collect();
        let pe_tensor = Tensor::from_slice(&pe_flat)
            .view([max_seq_len, hidden_dim])
            .unsqueeze(0) // [1, max_seq_len, hidden_dim]
            .to(device);
        
        Self { pe: pe_tensor }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.size()[1];
        x + &self.pe.narrow(1, 0, seq_len)
    }
}

/// Multi-Head Self-Attention
pub struct MultiHeadSelfAttention {
    num_heads: i64,
    head_dim: i64,
    hidden_dim: i64,
    
    w_query: nn::Linear,
    w_key: nn::Linear,
    w_value: nn::Linear,
    w_out: nn::Linear,
    
    dropout: f64,
    scale: f64,
}

impl MultiHeadSelfAttention {
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_heads: i64, dropout: f64) -> Self {
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be divisible by num_heads"
        );
        
        let head_dim = hidden_dim / num_heads;
        let scale = (head_dim as f64).sqrt();
        
        let w_query = nn::linear(vs / "w_query", hidden_dim, hidden_dim, Default::default());
        let w_key = nn::linear(vs / "w_key", hidden_dim, hidden_dim, Default::default());
        let w_value = nn::linear(vs / "w_value", hidden_dim, hidden_dim, Default::default());
        let w_out = nn::linear(vs / "w_out", hidden_dim, hidden_dim, Default::default());
        
        Self {
            num_heads,
            head_dim,
            hidden_dim,
            w_query,
            w_key,
            w_value,
            w_out,
            dropout,
            scale,
        }
    }
    
    fn split_heads(&self, x: &Tensor) -> Tensor {
        let sizes = x.size();
        let batch_size = sizes[0];
        let seq_len = sizes[1];
        
        x.view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .contiguous()
    }
    
    fn merge_heads(&self, x: &Tensor) -> Tensor {
        let sizes = x.size();
        let batch_size = sizes[0];
        let seq_len = sizes[2];
        
        x.transpose(1, 2)
            .contiguous()
            .view([batch_size, seq_len, self.hidden_dim])
    }
    
    /// Self-attention forward pass
    /// input: [B, T, H]
    /// mask: Optional attention mask
    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Linear projections
        let q = self.w_query.forward(input); // [B, T, H]
        let k = self.w_key.forward(input);   // [B, T, H]
        let v = self.w_value.forward(input); // [B, T, H]
        
        // Split into multiple heads
        let q = self.split_heads(&q); // [B, num_heads, T, head_dim]
        let k = self.split_heads(&k); // [B, num_heads, T, head_dim]
        let v = self.split_heads(&v); // [B, num_heads, T, head_dim]
        
        // Scaled dot-product attention
        let mut scores = q.matmul(&k.transpose(-2, -1)) / self.scale; // [B, num_heads, T, T]
        
        // Apply attention mask (for padding)
        if let Some(m) = mask {
            // Input mask: [B, T, T] for causal or [B, T] for padding
            // We need it to be [B, 1, T, T] for broadcasting with scores [B, num_heads, T, T]
            let mask_expanded = if m.size().len() == 2 {
                // Padding mask [B, T]
                m.unsqueeze(1)  // [B, 1, T]
                    .unsqueeze(1)  // [B, 1, 1, T]
                    .to_kind(Kind::Float)
            } else if m.size().len() == 3 {
                // Causal mask [B, T, T]
                m.unsqueeze(1)  // [B, 1, T, T]
                    .to_kind(Kind::Float)
            } else {
                panic!("Unexpected mask dimension: {:?}", m.size());
            };
            
            // Apply mask: valid=1.0, masked=0.0
            // mask_expanded is [B, 1, T, T], scores is [B, num_heads, T, T]
            // Broadcasting will handle the expansion automatically
            let neg_inf = Tensor::from(-1e9f64).to_device(scores.device());
            let inv_mask = 1.0 - &mask_expanded;
            
            // Properly apply mask: valid positions keep scores, masked positions get -inf
            scores = &scores * &mask_expanded + &neg_inf * inv_mask;
        }
        
        // Softmax
        let attn_weights = scores.softmax(-1, Kind::Float);
        
        // Apply dropout
        let attn_dropped = if self.dropout > 0.0 {
            attn_weights.dropout(self.dropout, true)
        } else {
            attn_weights.shallow_clone()
        };
        
        // Apply to values
        let context = attn_dropped.matmul(&v); // [B, num_heads, T, head_dim]
        
        // Merge heads
        let context = self.merge_heads(&context); // [B, T, H]
        
        // Output projection
        self.w_out.forward(&context)
    }
}

/// Cross-Attention (for decoder attending to encoder)
pub struct CrossAttention {
    num_heads: i64,
    head_dim: i64,
    hidden_dim: i64,
    
    w_query: nn::Linear,
    w_key: nn::Linear,
    w_value: nn::Linear,
    w_out: nn::Linear,
    
    dropout: f64,
    scale: f64,
}

impl CrossAttention {
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_heads: i64, dropout: f64) -> Self {
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be divisible by num_heads"
        );
        
        let head_dim = hidden_dim / num_heads;
        let scale = (head_dim as f64).sqrt();
        
        let w_query = nn::linear(vs / "w_query", hidden_dim, hidden_dim, Default::default());
        let w_key = nn::linear(vs / "w_key", hidden_dim, hidden_dim, Default::default());
        let w_value = nn::linear(vs / "w_value", hidden_dim, hidden_dim, Default::default());
        let w_out = nn::linear(vs / "w_out", hidden_dim, hidden_dim, Default::default());
        
        Self {
            num_heads,
            head_dim,
            hidden_dim,
            w_query,
            w_key,
            w_value,
            w_out,
            dropout,
            scale,
        }
    }
    
    fn split_heads(&self, x: &Tensor) -> Tensor {
        let sizes = x.size();
        let batch_size = sizes[0];
        let seq_len = sizes[1];
        
        x.view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .contiguous()
    }
    
    fn merge_heads(&self, x: &Tensor) -> Tensor {
        let sizes = x.size();
        let batch_size = sizes[0];
        let seq_len = sizes[2];
        
        x.transpose(1, 2)
            .contiguous()
            .view([batch_size, seq_len, self.hidden_dim])
    }
    
    /// Cross-attention forward pass
    /// query: [B, Tt, H] (from decoder)
    /// key: [B, Ts, H] (from encoder)
    /// value: [B, Ts, H] (from encoder)
    /// mask: Optional encoder mask
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Tensor {
        
        // Linear projections
        let q = self.w_query.forward(query); // [B, Tt, H]
        let k = self.w_key.forward(key);     // [B, Ts, H]
        let v = self.w_value.forward(value); // [B, Ts, H]
        
        // Split into multiple heads
        let q = self.split_heads(&q); // [B, num_heads, Tt, head_dim]
        let k = self.split_heads(&k); // [B, num_heads, Ts, head_dim]
        let v = self.split_heads(&v); // [B, num_heads, Ts, head_dim]
        
        // Scaled dot-product attention
        let mut scores = q.matmul(&k.transpose(-2, -1)) / self.scale; // [B, num_heads, Tt, Ts]
        
        // Apply encoder mask
        if let Some(m) = mask {
            let mask_expanded = m.unsqueeze(1).unsqueeze(1); // [B, 1, 1, Ts]
            
            // Convert bool mask to float (1.0 for valid, 0.0 for masked)
            let mask_float = mask_expanded.to_kind(Kind::Float);
            
            // Create inverse mask (1.0 for masked positions)
            let inv_mask = 1.0 - &mask_float;
            
            let neg_inf = Tensor::from(-1e9f64).to_device(scores.device());
            scores = &scores * &mask_float + &neg_inf * inv_mask;
        }
        
        // Softmax
        let attn_weights = scores.softmax(-1, Kind::Float);
        
        // Apply dropout
        let attn_dropped = if self.dropout > 0.0 {
            attn_weights.dropout(self.dropout, true)
        } else {
            attn_weights.shallow_clone()
        };
        
        // Apply to values
        let context = attn_dropped.matmul(&v); // [B, num_heads, Tt, head_dim]
        
        // Merge heads
        let context = self.merge_heads(&context); // [B, Tt, H]
        
        // Output projection
        self.w_out.forward(&context)
    }
}

/// Position-wise Feed-Forward Network
pub struct FeedForward {
    linear1: nn::Linear,
    linear2: nn::Linear,
    dropout: f64,
}

impl FeedForward {
    pub fn new(vs: &nn::Path, hidden_dim: i64, ff_dim: i64, dropout: f64) -> Self {
        let linear1 = nn::linear(vs / "linear1", hidden_dim, ff_dim, Default::default());
        let linear2 = nn::linear(vs / "linear2", ff_dim, hidden_dim, Default::default());
        
        Self {
            linear1,
            linear2,
            dropout,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = self.linear1.forward(x).relu();
        let hidden_dropped = if self.dropout > 0.0 {
            hidden.dropout(self.dropout, true)
        } else {
            hidden
        };
        self.linear2.forward(&hidden_dropped)
    }
}

/// Transformer Encoder Layer
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadSelfAttention,
    feed_forward: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    dropout: f64,
}

impl TransformerEncoderLayer {
    pub fn new(
        vs: &nn::Path,
        hidden_dim: i64,
        num_heads: i64,
        ff_dim: i64,
        dropout: f64,
    ) -> Self {
        let self_attn = MultiHeadSelfAttention::new(
            &(vs / "self_attn"),
            hidden_dim,
            num_heads,
            dropout,
        );
        
        let feed_forward = FeedForward::new(&(vs / "ff"), hidden_dim, ff_dim, dropout);
        
        let norm1 = nn::layer_norm(vs / "norm1", vec![hidden_dim], Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![hidden_dim], Default::default());
        
        Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            dropout,
        }
    }
    
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Self-attention with residual and layer norm
        let attn_out = self.self_attn.forward(x, mask);
        let attn_out = if self.dropout > 0.0 {
            attn_out.dropout(self.dropout, true)
        } else {
            attn_out
        };
        let x = self.norm1.forward(&(x + &attn_out));
        
        // Feed-forward with residual and layer norm
        let ff_out = self.feed_forward.forward(&x);
        let ff_out = if self.dropout > 0.0 {
            ff_out.dropout(self.dropout, true)
        } else {
            ff_out
        };
        self.norm2.forward(&(x + &ff_out))
    }
}

/// Transformer Decoder Layer
pub struct TransformerDecoderLayer {
    self_attn: MultiHeadSelfAttention,
    cross_attn: CrossAttention,
    feed_forward: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
    dropout: f64,
}

impl TransformerDecoderLayer {
    pub fn new(
        vs: &nn::Path,
        hidden_dim: i64,
        num_heads: i64,
        ff_dim: i64,
        dropout: f64,
    ) -> Self {
        let self_attn = MultiHeadSelfAttention::new(
            &(vs / "self_attn"),
            hidden_dim,
            num_heads,
            dropout,
        );
        
        let cross_attn = CrossAttention::new(
            &(vs / "cross_attn"),
            hidden_dim,
            num_heads,
            dropout,
        );
        
        let feed_forward = FeedForward::new(&(vs / "ff"), hidden_dim, ff_dim, dropout);
        
        let norm1 = nn::layer_norm(vs / "norm1", vec![hidden_dim], Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![hidden_dim], Default::default());
        let norm3 = nn::layer_norm(vs / "norm3", vec![hidden_dim], Default::default());
        
        Self {
            self_attn,
            cross_attn,
            feed_forward,
            norm1,
            norm2,
            norm3,
            dropout,
        }
    }
    
    pub fn forward(
        &self,
        x: &Tensor,
        encoder_output: &Tensor,
        decoder_mask: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
    ) -> Tensor {
        // Self-attention on decoder with causal mask
        let self_attn_out = self.self_attn.forward(x, decoder_mask);
        let self_attn_out = if self.dropout > 0.0 {
            self_attn_out.dropout(self.dropout, true)
        } else {
            self_attn_out
        };
        let x = self.norm1.forward(&(x + &self_attn_out));
        
        // Cross-attention to encoder
        let cross_attn_out = self.cross_attn.forward(&x, encoder_output, encoder_output, encoder_mask);
        let cross_attn_out = if self.dropout > 0.0 {
            cross_attn_out.dropout(self.dropout, true)
        } else {
            cross_attn_out
        };
        let x = self.norm2.forward(&(x + &cross_attn_out));
        
        // Feed-forward
        let ff_out = self.feed_forward.forward(&x);
        let ff_out = if self.dropout > 0.0 {
            ff_out.dropout(self.dropout, true)
        } else {
            ff_out
        };
        self.norm3.forward(&(x + &ff_out))
    }
}

/// Create causal attention mask for decoder
pub fn create_causal_mask(seq_len: i64, device: tch::Device) -> Tensor {
    let mut mask = vec![vec![1i64; seq_len as usize]; seq_len as usize];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i as usize][j as usize] = 0;
        }
    }
    let mask_flat: Vec<i64> = mask.iter().flat_map(|row| row.clone()).collect();
    Tensor::from_slice(&mask_flat)
        .view([seq_len, seq_len])
        .unsqueeze(0)
        .to(device)
}

/// Create padding mask
pub fn create_padding_mask(lengths: &[i64], max_len: i64, device: tch::Device) -> Tensor {
    let batch_size = lengths.len() as i64;
    let mut mask = vec![vec![1i64; max_len as usize]; batch_size as usize];
    
    for (b, &len) in lengths.iter().enumerate() {
        for i in len..max_len {
            mask[b][i as usize] = 0;
        }
    }
    
    let mask_flat: Vec<i64> = mask.iter().flat_map(|row| row.clone()).collect();
    Tensor::from_slice(&mask_flat)
        .view([batch_size, max_len])
        .to(device)
}
