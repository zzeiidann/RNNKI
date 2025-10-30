// src/attention.rs
use tch::{nn, nn::Module, Tensor, Kind};

/// Multi-Head Attention (Scaled Dot-Product)
pub struct MultiHeadAttention {
    num_heads: i64,
    head_dim: i64,
    hidden_dim: i64,
    
    // Projections untuk Q, K, V
    w_query: nn::Linear,
    w_key: nn::Linear,
    w_value: nn::Linear,
    
    // Output projection
    w_out: nn::Linear,
    
    dropout: f64,
    scale: f64,
}

impl MultiHeadAttention {
    /// hidden_dim: dimensi total (harus habis dibagi num_heads)
    /// num_heads: jumlah attention heads
    /// dropout: dropout rate untuk attention weights
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_heads: i64, dropout: f64) -> Self {
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be divisible by num_heads"
        );
        
        let head_dim = hidden_dim / num_heads;
        let scale = (head_dim as f64).sqrt();
        
        // Linear projections untuk Q, K, V
        let w_query = nn::linear(vs / "w_query", hidden_dim, hidden_dim, Default::default());
        let w_key = nn::linear(vs / "w_key", hidden_dim, hidden_dim, Default::default());
        let w_value = nn::linear(vs / "w_value", hidden_dim, hidden_dim, Default::default());
        
        // Output projection
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
    
    /// Split tensor menjadi multiple heads
    /// input: [B, T, H] -> output: [B, num_heads, T, head_dim]
    fn split_heads(&self, x: &Tensor) -> Tensor {
        let sizes = x.size();
        let batch_size = sizes[0];
        let seq_len = sizes[1];
        
        x.view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .contiguous()
    }
    
    /// Merge multiple heads kembali
    /// input: [B, num_heads, T, head_dim] -> output: [B, T, H]
    fn merge_heads(&self, x: &Tensor) -> Tensor {
        let sizes = x.size();
        let batch_size = sizes[0];
        let seq_len = sizes[2];
        
        x.transpose(1, 2)
            .contiguous()
            .view([batch_size, seq_len, self.hidden_dim])
    }
    
    /// Forward pass untuk multi-head attention
    /// query: [B, H] (decoder hidden state)
    /// keys: [B, T, H] (encoder outputs)
    /// values: [B, T, H] (encoder outputs, bisa sama dengan keys)
    /// mask: Option<[B, T]> (padding mask)
    /// 
    /// Returns: (context [B, H], attention_weights [B, num_heads, T])
    pub fn forward(
        &self,
        query: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        let batch_size = keys.size()[0];
        let seq_len = keys.size()[1];
        
        // Expand query dari [B, H] -> [B, 1, H] untuk kompatibilitas
        let query_expanded = query.unsqueeze(1);
        
        // Linear projections
        let q = self.w_query.forward(&query_expanded); // [B, 1, H]
        let k = self.w_key.forward(keys);               // [B, T, H]
        let v = self.w_value.forward(values);           // [B, T, H]
        
        // Split into multiple heads
        let q = self.split_heads(&q); // [B, num_heads, 1, head_dim]
        let k = self.split_heads(&k); // [B, num_heads, T, head_dim]
        let v = self.split_heads(&v); // [B, num_heads, T, head_dim]
        
        // Scaled dot-product attention
        // scores = (Q @ K^T) / sqrt(d_k)
        let mut scores = q.matmul(&k.transpose(-2, -1)) / self.scale; // [B, num_heads, 1, T]
        
        // Apply mask jika ada
        if let Some(m) = mask {
            // Normalize mask ke [B, T]
            let m_normalized = if m.size()[0] == batch_size && m.size()[1] == seq_len {
                m.shallow_clone()
            } else if m.size()[0] == seq_len && m.size()[1] == batch_size {
                m.transpose(0, 1)
            } else {
                m.view([batch_size, seq_len])
            };
            
            // Convert to float mask (1 = valid, 0 = masked)
            let mask_float = if m_normalized.kind() == Kind::Bool {
                m_normalized.to_kind(Kind::Float)
            } else {
                m_normalized.ne(0).to_kind(Kind::Float)
            };
            
            // Expand mask untuk broadcasting: [B, 1, 1, T]
            let mask_expanded = mask_float.unsqueeze(1).unsqueeze(2);
            
            // Apply mask: masked positions get -1e9
            let neg_inf = Tensor::from(-1e9f64).to_device(scores.device());
            let inv_mask = mask_expanded.neg() + 1.0; // 1 - mask (inverted)
            scores = &scores * &mask_expanded + &neg_inf * inv_mask;
        }
        
        // Softmax over sequence dimension
        let attn_weights = scores.softmax(-1, Kind::Float); // [B, num_heads, 1, T]
        
        // Apply dropout jika training
        let attn_dropped = if self.dropout > 0.0 {
            attn_weights.dropout(self.dropout, true)
        } else {
            attn_weights.shallow_clone()
        };
        
        // Apply attention to values
        let context = attn_dropped.matmul(&v); // [B, num_heads, 1, head_dim]
        
        // Merge heads
        let context = self.merge_heads(&context); // [B, 1, H]
        let context = context.squeeze_dim(1);     // [B, H]
        
        // Output projection
        let output = self.w_out.forward(&context.unsqueeze(1)).squeeze_dim(1); // [B, H]
        
        // Return context dan average attention weights across heads
        let attn_avg = attn_weights.mean_dim(&[1i64][..], false, Kind::Float).squeeze_dim(1); // [B, T]
        
        (output, attn_avg)
    }
}

/// Bahdanau (Additive) Attention - kept for backward compatibility
pub struct BahdanauAttention {
    w_query: nn::Linear,
    w_key: nn::Linear,
    v: nn::Linear,
}

impl BahdanauAttention {
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        let w_query = nn::linear(vs / "w_query", hidden_dim, hidden_dim, Default::default());
        let w_key = nn::linear(vs / "w_key", hidden_dim, hidden_dim, Default::default());
        let v = nn::linear(vs / "v", hidden_dim, 1, Default::default());
        Self { w_query, w_key, v }
    }

    pub fn forward(&self, query: &Tensor, keys_in: &Tensor, mask: Option<&Tensor>) -> (Tensor, Tensor) {
        let b = query.size()[0];

        let keys = if keys_in.dim() == 3 && keys_in.size()[0] != b && keys_in.size()[1] == b {
            keys_in.transpose(0, 1)
        } else {
            keys_in.shallow_clone()
        };
        let seq_len = keys.size()[1];

        let q_proj = self.w_query.forward(query).unsqueeze(1);
        let k_proj = self.w_key.forward(&keys);

        let mut e = (q_proj + k_proj).tanh().apply(&self.v).squeeze_dim(-1);

        if let Some(m_in) = mask {
            let m_bt = if m_in.dim() == 2 {
                if m_in.size()[0] == b && m_in.size()[1] == seq_len {
                    m_in.shallow_clone()
                } else if m_in.size()[0] == seq_len && m_in.size()[1] == b {
                    m_in.transpose(0, 1)
                } else {
                    m_in.view([b, seq_len])
                }
            } else {
                m_in.view([b, seq_len])
            };

            let m_f = if m_bt.kind() == Kind::Bool {
                m_bt.to_kind(Kind::Float)
            } else {
                m_bt.ne(0).to_kind(Kind::Float)
            };
            let neg = Tensor::from(-1e30f64).to_device(e.device());
            let inv = m_f.neg() + 1.0;
            e = &e * &m_f + &neg * inv;
        }

        let attn = e.softmax(-1, Kind::Float);
        let context = attn.unsqueeze(1).bmm(&keys).squeeze_dim(1);

        (context, attn)
    }
}

/// Scaled Dot-Product Attention (single head version)
pub struct ScaledDotProductAttention {
    scale: f64,
}

impl ScaledDotProductAttention {
    pub fn new(hidden_dim: i64) -> Self {
        let scale = (hidden_dim as f64).sqrt();
        Self { scale }
    }

    pub fn forward(&self, query: &Tensor, keys_in: &Tensor, mask: Option<&Tensor>) -> (Tensor, Tensor) {
        let b = query.size()[0];

        let keys = if keys_in.dim() == 3 && keys_in.size()[0] != b && keys_in.size()[1] == b {
            keys_in.transpose(0, 1)
        } else {
            keys_in.shallow_clone()
        };
        let seq_len = keys.size()[1];

        let q = query.unsqueeze(1);
        let mut scores = q.bmm(&keys.transpose(1, 2)) / self.scale;
        scores = scores.squeeze_dim(1);

        if let Some(m_in) = mask {
            let m_bt = if m_in.dim() == 2 {
                if m_in.size()[0] == b && m_in.size()[1] == seq_len {
                    m_in.shallow_clone()
                } else if m_in.size()[0] == seq_len && m_in.size()[1] == b {
                    m_in.transpose(0, 1)
                } else {
                    m_in.view([b, seq_len])
                }
            } else {
                m_in.view([b, seq_len])
            };

            let m_f = if m_bt.kind() == Kind::Bool {
                m_bt.to_kind(Kind::Float)
            } else {
                m_bt.ne(0).to_kind(Kind::Float)
            };
            let neg = Tensor::from(-1e30f64).to_device(scores.device());
            let inv = m_f.neg() + 1.0;
            scores = &scores * &m_f + &neg * inv;
        }

        let attn = scores.softmax(-1, Kind::Float);
        let context = attn.unsqueeze(1).bmm(&keys).squeeze_dim(1);

        (context, attn)
    }
}