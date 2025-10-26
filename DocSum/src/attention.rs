// src/attention.rs
use tch::{nn, nn::Module, Tensor, Kind};

/// Bahdanau (Additive) Attention
pub struct BahdanauAttention {
    w_query: nn::Linear, // query (B,Hq) -> A  ; Hq == 2H, A == 2H (simplify)
    w_key:   nn::Linear, // key   (B,T,Hk) -> A ; Hk == 2H
    v:       nn::Linear, // (B,T,A) -> (B,T,1)
}

impl BahdanauAttention {
    /// hidden_dim di sini = dim encoder bi-dir (2H)
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        let w_query = nn::linear(vs / "w_query", hidden_dim, hidden_dim, Default::default());
        let w_key   = nn::linear(vs / "w_key",   hidden_dim, hidden_dim, Default::default());
        let v       = nn::linear(vs / "v",       hidden_dim, 1,           Default::default());
        Self { w_query, w_key, v }
    }

    /// query: [B, Hq] (Hq=2H), keys_in: [B,T,H] atau [T,B,H], mask: [B,T] atau [T,B]
    /// return: (context [B,H], attention_weights [B,T])
    pub fn forward(&self, query: &Tensor, keys_in: &Tensor, mask: Option<&Tensor>) -> (Tensor, Tensor) {
        let b = query.size()[0];

        // --- Normalisasi keys ke [B, T, H] ---
        let keys = if keys_in.dim() == 3 && keys_in.size()[0] != b && keys_in.size()[1] == b {
            // kemungkinan datang sebagai [T,B,H] -> [B,T,H]
            keys_in.transpose(0, 1)
        } else {
            keys_in.shallow_clone()
        };
        let seq_len = keys.size()[1];

        // --- Proyeksi: broadcast aman (B,1,A) + (B,T,A) ---
        let q_proj = self.w_query.forward(query).unsqueeze(1); // [B,1,A]
        let k_proj = self.w_key.forward(&keys);                // [B,T,A]

        // e = v^T tanh(Wq q + Wk k_i) -> [B,T]
        let mut e = (q_proj + k_proj).tanh().apply(&self.v).squeeze_dim(-1); // [B,T]

        // --- Masking aman (pad -> -1e30) + normalisasi mask ke [B,T] ---
        if let Some(m_in) = mask {
            let m_bt = if m_in.dim() == 2 {
                // m_in bisa [B,T] atau [T,B]
                if m_in.size()[0] == b && m_in.size()[1] == seq_len {
                    m_in.shallow_clone()
                } else if m_in.size()[0] == seq_len && m_in.size()[1] == b {
                    m_in.transpose(0, 1)
                } else {
                    // fallback: coba reshape (jarang terjadi)
                    m_in.view([b, seq_len])
                }
            } else {
                // fallback lain: broadcast ke [B,T]
                m_in.view([b, seq_len])
            };

            // ubah ke float 0/1
            let m_f = if m_bt.kind() == Kind::Bool { m_bt.to_kind(Kind::Float) } else { m_bt.ne(0).to_kind(Kind::Float) };
            let neg = Tensor::from(-1e30f64).to_device(e.device()); // scalar di device yang sama
            let inv = m_f.neg() + 1.0; // 1 - m
            // e_masked = e * m + neg * (1 - m)
            e = &e * &m_f + &neg * inv;
        }

        // softmax over T
        let attn = e.softmax(-1, Kind::Float); // [B,T]

        // Context: Σ α_i * h_i -> [B,H]
        let context = attn.unsqueeze(1).bmm(&keys).squeeze_dim(1);

        (context, attn)
    }
}

/// (Opsional) Scaled Dot-Product Attention versi batch-first
pub struct ScaledDotProductAttention {
    scale: f64,
}

impl ScaledDotProductAttention {
    pub fn new(hidden_dim: i64) -> Self {
        let scale = (hidden_dim as f64).sqrt();
        Self { scale }
    }

    /// query: [B,H], keys_in: [B,T,H] atau [T,B,H], mask: [B,T] atau [T,B]
    pub fn forward(&self, query: &Tensor, keys_in: &Tensor, mask: Option<&Tensor>) -> (Tensor, Tensor) {
        let b = query.size()[0];

        // keys -> [B,T,H]
        let keys = if keys_in.dim() == 3 && keys_in.size()[0] != b && keys_in.size()[1] == b {
            keys_in.transpose(0, 1)
        } else {
            keys_in.shallow_clone()
        };
        let seq_len = keys.size()[1];

        // skor: (Q K^T) / sqrt(d_k)
        let q = query.unsqueeze(1); // [B,1,H]
        let mut scores = q.bmm(&keys.transpose(1, 2)) / self.scale; // [B,1,T]
        scores = scores.squeeze_dim(1); // [B,T]

        // mask -> [B,T]
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

            let m_f = if m_bt.kind() == Kind::Bool { m_bt.to_kind(Kind::Float) } else { m_bt.ne(0).to_kind(Kind::Float) };
            let neg = Tensor::from(-1e30f64).to_device(scores.device());
            let inv = m_f.neg() + 1.0; // 1 - m
            scores = &scores * &m_f + &neg * inv;
        }

        let attn = scores.softmax(-1, Kind::Float);                // [B,T]
        let context = attn.unsqueeze(1).bmm(&keys).squeeze_dim(1); // [B,H]

        (context, attn)
    }
}
