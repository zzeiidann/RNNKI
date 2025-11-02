// src/model.rs
use tch::{nn, nn::Module, Tensor};
use crate::transformer::{
    PositionalEncoding, TransformerEncoderLayer, TransformerDecoderLayer,
    create_causal_mask,
};

/// Transformer Seq2Seq model with Multi-Head Attention
pub struct TransformerSeq2Seq {
    embedding: nn::Embedding,
    embedding_to_hidden: nn::Linear,  // Project embedding to hidden dim
    encoder_pos_encoding: PositionalEncoding,
    decoder_pos_encoding: PositionalEncoding,
    
    // Encoder layers
    encoder_layers: Vec<TransformerEncoderLayer>,
    
    // Decoder layers
    decoder_layers: Vec<TransformerDecoderLayer>,
    
    // Output projection
    output_projection: nn::Linear,
    
    hidden_dim: i64,
    num_layers: i64,
    dropout: f64,
    device: tch::Device,
}

impl TransformerSeq2Seq {
    pub fn new(
        vs: &nn::Path,
        vocab_size: i64,
        embedding_dim: i64,
        hidden_dim: i64,
        num_layers: i64,
        num_heads: i64,
        ff_dim: i64,
        dropout: f64,
        device: tch::Device,
    ) -> Self {
        let embedding = nn::embedding(
            vs / "embedding",
            vocab_size,
            embedding_dim,
            Default::default(),
        );
        
        // Positional encoding (use hidden_dim, not embedding_dim)
        let encoder_pos_encoding = PositionalEncoding::new(512, hidden_dim, device);
        let decoder_pos_encoding = PositionalEncoding::new(512, hidden_dim, device);
        
        // Project embedding to hidden dimension
        let embedding_to_hidden = nn::linear(
            vs / "embedding_to_hidden",
            embedding_dim,
            hidden_dim,
            Default::default(),
        );
        
        // Encoder and decoder layers
        let mut encoder_layers = Vec::new();
        for i in 0..num_layers {
            let layer = TransformerEncoderLayer::new(
                &(vs / format!("encoder_layer_{}", i)),
                hidden_dim,
                num_heads,
                ff_dim,
                dropout,
            );
            encoder_layers.push(layer);
        }
        
        // Decoder layers
        let mut decoder_layers = Vec::new();
        for i in 0..num_layers {
            let layer = TransformerDecoderLayer::new(
                &(vs / format!("decoder_layer_{}", i)),
                hidden_dim,
                num_heads,
                ff_dim,
                dropout,
            );
            decoder_layers.push(layer);
        }
        
        // Output projection
        let output_projection = nn::linear(
            vs / "output",
            hidden_dim,
            vocab_size,
            Default::default(),
        );
        
        Self {
            embedding,
            embedding_to_hidden,
            encoder_pos_encoding,
            decoder_pos_encoding,
            encoder_layers,
            decoder_layers,
            output_projection,
            hidden_dim,
            num_layers,
            dropout,
            device,
        }
    }
    
    /// Encode input sequence
    /// encoder_input: [B, Ts]
    /// encoder_mask: [B, Ts] (1 = valid, 0 = pad)
    pub fn encode(&self, encoder_input: &Tensor, encoder_mask: Option<&Tensor>) -> Tensor {
        // Embedding
        let mut x = self.embedding.forward(encoder_input); // [B, Ts, E]
        
        // Project embedding to hidden dimension
        x = self.embedding_to_hidden.forward(&x); // [B, Ts, H]
        
        // Positional encoding
        x = self.encoder_pos_encoding.forward(&x); // [B, Ts, H]
        
        // Encoder layers
        for layer in &self.encoder_layers {
            x = layer.forward(&x, encoder_mask);
        }
        
        x // [B, Ts, H]
    }
    
    /// Decode one step with attention to encoder
    /// decoder_input: [B, Tt]
    /// encoder_output: [B, Ts, H]
    /// encoder_mask: [B, Ts]
    pub fn decode_step(
        &self,
        decoder_input: &Tensor,
        encoder_output: &Tensor,
        encoder_mask: Option<&Tensor>,
    ) -> Tensor {
        // Embedding
        let mut x = self.embedding.forward(decoder_input); // [B, Tt, E]
        
        // Project embedding to hidden dimension
        x = self.embedding_to_hidden.forward(&x); // [B, Tt, H]
        
        // Positional encoding
        x = self.decoder_pos_encoding.forward(&x); // [B, Tt, H]
        
        // Get causal mask for decoder with correct batch size
        let batch_size = decoder_input.size()[0];
        let seq_len = decoder_input.size()[1];
        let causal_mask = self.create_causal_mask_for_batch(batch_size, seq_len);
        
        // Decoder layers with cross-attention
        for layer in &self.decoder_layers {
            x = layer.forward(&x, encoder_output, Some(&causal_mask), encoder_mask);
        }
        
        // Output projection
        let logits = self.output_projection.forward(&x); // [B, Tt, V]
        logits
    }
    
    /// Create causal mask with correct batch dimension
    /// Returns: [B, T, T] causal mask (lower triangular)
    fn create_causal_mask_for_batch(&self, batch_size: i64, seq_len: i64) -> Tensor {
        // Create single causal mask [T, T]
        let mut causal_mask = vec![vec![1i64; seq_len as usize]; seq_len as usize];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                causal_mask[i as usize][j as usize] = 0;
            }
        }
        
        // Flatten and convert to tensor [T, T]
        let mask_flat: Vec<i64> = causal_mask.iter().flat_map(|row| row.clone()).collect();
        let mask_t = Tensor::from_slice(&mask_flat)
            .view([seq_len, seq_len])
            .to_kind(tch::Kind::Float)
            .to(self.device);
        
        // Repeat for batch: stack mask batch_size times
        let mut masks = Vec::new();
        for _ in 0..batch_size {
            masks.push(mask_t.shallow_clone());
        }
        Tensor::stack(&masks, 0)
    }
    
    /// Forward pass for training (teacher forcing)
    pub fn forward(
        &self,
        encoder_input: &Tensor,
        decoder_input: &Tensor,
        encoder_mask: Option<&Tensor>,
    ) -> Tensor {
        // Encode
        let encoder_output = self.encode(encoder_input, encoder_mask);
        
        // Decode with teacher forcing
        self.decode_step(decoder_input, &encoder_output, encoder_mask)
    }
    
    /// Generate summary step-by-step (greedy decoding)
    pub fn generate_greedy(
        &self,
        encoder_input: &Tensor,
        encoder_mask: Option<&Tensor>,
        max_len: i64,
        eos_token_id: i64,
        sos_token_id: i64,
    ) -> Tensor {
        let device = encoder_input.device();
        
        // Encode
        let encoder_output = self.encode(encoder_input, encoder_mask);
        
        // Start with SOS token
        let mut generated = vec![sos_token_id];
        
        for _ in 0..max_len {
            // Convert to tensor
            let decoder_input_ids: Vec<i64> = generated.iter().copied().collect();
            let decoder_input = Tensor::from_slice(&decoder_input_ids)
                .view([1, generated.len() as i64])
                .to(device);
            
            // Decode
            let logits = self.decode_step(&decoder_input, &encoder_output, encoder_mask);
            
            // Get last token prediction
            let last_logits = logits.select(1, (logits.size()[1] - 1) as i64);
            let next_token = last_logits.argmax(-1, false);
            let next_token_id = i64::try_from(&next_token).unwrap_or(eos_token_id);
            
            generated.push(next_token_id);
            
            if next_token_id == eos_token_id {
                break;
            }
        }
        
        Tensor::from_slice(&generated)
            .view([1, generated.len() as i64])
            .to(device)
    }
}

// Legacy AttentionRNN for backward compatibility (kept for reference)
pub struct AttentionRNN;

impl AttentionRNN {
    pub fn new(
        _vs: &nn::Path,
        _vocab_size: i64,
        _embedding_dim: i64,
        _hidden_dim: i64,
        _num_layers: i64,
        _dropout: f64,
    ) -> Self {
        panic!("AttentionRNN is deprecated. Use TransformerSeq2Seq instead.");
    }
}
