"""
Transformer Facet - Attention-based context processor for cognitive architectures.

Unlike CharmNetwork (LSTM/GRU for temporal affect), TransformerFacet uses
self-attention to process context where RELATIONSHIPS matter more than ORDER.

Use cases in cognitive architectures:
- Social context parsing (who said what, to whom, about whom)
- Intent detection (what does the user want)
- Pronoun/reference resolution ("it", "they", "that thing")
- Multi-party conversation tracking
- Scene understanding (which entities are relevant)

The key insight: Attention lets the model CHOOSE what to focus on,
rather than processing everything sequentially like LSTM.

"The cat sat on the mat because it was tired"
Attention can directly connect "it" to "cat" without processing
every word in between.

Author: Commander Spock + Cadet Caity
Date: December 19, 2025
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np


@dataclass
class TransformerOutput:
    """Output from Transformer Facet processing."""

    # Processed context embedding (can feed to downstream facets)
    context_embedding: List[float]

    # Attention weights for explainability (which tokens attended to which)
    attention_weights: List[List[float]]

    # Optional: classification output if configured
    classification: Optional[Dict[str, float]] = None

    # Token-level outputs (for each input token)
    token_outputs: Optional[List[List[float]]] = None

    # Which tokens were most attended to (for debugging/viz)
    top_attended_tokens: Optional[List[Tuple[int, float]]] = None

    # Processing metadata
    num_tokens: int = 0
    processing_time_ms: float = 0.0


class TransformerFacet:
    """
    Transformer-based context processor for cognitive architectures.

    This facet uses self-attention to understand context where relationships
    between elements matter more than strict sequential order.

    Unlike LLM facets (which call external APIs), this runs a small local
    transformer model optimized for specific cognitive tasks.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        max_seq_len: int = 128,
        vocab_size: int = 10000,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize Transformer Facet.

        Args:
            embed_dim: Embedding dimension (default 64 for efficiency)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size for embedding
            num_classes: If set, adds classification head
            dropout: Dropout rate (0.0 for inference)
            checkpoint_path: Optional path to trained weights
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        # Build model
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")

        self._build_model(dropout)

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Move to device
        self.model.to(self.device)
        self.model.eval()  # Inference mode

        # Execution statistics
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0

        # Execution lock for thread safety
        self.execution_lock = asyncio.Lock()

        print(f"[Transformer Facet] Initialized on {self.device}")
        print(f"[Transformer Facet] {self._count_parameters():,} parameters")

    def _build_model(self, dropout: float):
        """Build the transformer model architecture."""

        # Token embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Positional encoding (sinusoidal)
        self.pos_encoding = self._create_positional_encoding()

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Output projection (context embedding)
        self.context_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Optional classification head
        if self.num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, self.ff_dim),
                nn.GELU(),
                nn.Linear(self.ff_dim, self.num_classes)
            )
        else:
            self.classifier = None

        # Wrap in ModuleList for easy access
        self.model = nn.ModuleList([
            self.embedding,
            self.transformer,
            self.context_proj
        ])
        if self.classifier:
            self.model.append(self.classifier)

    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(self.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() *
            -(np.log(10000.0) / self.embed_dim)
        )

        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:self.embed_dim // 2])

        return pe.unsqueeze(0)  # (1, max_seq, embed)

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _load_checkpoint(self, path: str):
        """Load trained weights from checkpoint."""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[Transformer Facet] Loaded checkpoint: {path}")
        except Exception as e:
            print(f"[Transformer Facet] Failed to load checkpoint: {e}")

    def _compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for visualization.

        Note: PyTorch's TransformerEncoder doesn't expose attention weights
        directly, so we compute them manually for the first layer.
        """
        # Simple self-attention computation for visualization
        # Q = K = V = x
        d_k = x.shape[-1]
        scores = torch.bmm(x, x.transpose(1, 2)) / np.sqrt(d_k)
        weights = torch.softmax(scores, dim=-1)
        return weights

    async def process(
        self,
        token_ids: List[int],
        text: Optional[str] = None,
        return_attention: bool = True
    ) -> TransformerOutput:
        """
        Process token sequence through transformer.

        Args:
            token_ids: List of token IDs (from tokenizer)
            text: Optional raw text (for logging/debugging)
            return_attention: Whether to compute attention weights

        Returns:
            TransformerOutput with context embedding and attention weights
        """
        async with self.execution_lock:
            start_time = time.time()

            # Convert to tensor
            tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            seq_len = tokens.shape[1]

            # Truncate if too long
            if seq_len > self.max_seq_len:
                tokens = tokens[:, :self.max_seq_len]
                seq_len = self.max_seq_len

            with torch.no_grad():
                # Embedding + positional encoding
                x = self.embedding(tokens)  # (1, seq, embed)
                x = x + self.pos_encoding[:, :seq_len, :].to(self.device)

                # Compute attention weights before transformer (for viz)
                attn_weights = None
                if return_attention:
                    attn_weights = self._compute_attention_weights(x)

                # Transformer forward pass
                encoded = self.transformer(x)  # (1, seq, embed)

                # Context embedding (mean pooling over sequence)
                context = encoded.mean(dim=1)  # (1, embed)
                context = self.context_proj(context)

                # Optional classification
                classification = None
                if self.classifier:
                    logits = self.classifier(context)
                    probs = torch.softmax(logits, dim=-1)
                    classification = {
                        str(i): float(probs[0, i])
                        for i in range(probs.shape[1])
                    }

                # Find top attended tokens
                top_attended = None
                if attn_weights is not None:
                    # Average attention received by each token
                    avg_attention = attn_weights[0].mean(dim=0)  # (seq,)
                    top_k = min(5, seq_len)
                    top_indices = torch.topk(avg_attention, top_k).indices.tolist()
                    top_attended = [
                        (idx, float(avg_attention[idx]))
                        for idx in top_indices
                    ]

            # Record stats
            elapsed = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += elapsed
            self.last_execution_time = elapsed

            return TransformerOutput(
                context_embedding=context[0].cpu().tolist(),
                attention_weights=(
                    attn_weights[0].cpu().tolist()
                    if attn_weights is not None else []
                ),
                classification=classification,
                token_outputs=encoded[0].cpu().tolist(),
                top_attended_tokens=top_attended,
                num_tokens=seq_len,
                processing_time_ms=elapsed * 1000
            )

    async def process_text(
        self,
        text: str,
        tokenizer: Optional[Any] = None
    ) -> TransformerOutput:
        """
        Process raw text through transformer.

        Args:
            text: Raw text to process
            tokenizer: Optional tokenizer (uses simple word tokenization if None)

        Returns:
            TransformerOutput
        """
        if tokenizer:
            token_ids = tokenizer.encode(text)
        else:
            # Simple word-based tokenization (for demos)
            words = text.lower().split()
            # Hash words to token IDs (simple approach)
            token_ids = [hash(word) % self.vocab_size for word in words]

        return await self.process(token_ids, text=text)

    def reset_state(self):
        """
        Reset facet state.

        Note: Transformer is stateless (no hidden state like LSTM),
        so this just resets statistics.
        """
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = 0.0
        print("[Transformer Facet] State reset")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.execution_count,
            'total_tokens': 0,  # Neural network, not LLM
            'avg_tokens': 0,
            'total_time': self.total_execution_time,
            'avg_time': (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0
            ),
            'last_tokens': 0,
            'last_time': self.last_execution_time
        }

    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage (always 0 - local neural computation)."""
        return {
            'last_tokens': 0,
            'total_tokens': 0,
            'execution_count': self.execution_count,
            'avg_tokens': 0
        }


# Pre-configured facet variants for common cognitive tasks
class SocialContextFacet(TransformerFacet):
    """
    Transformer configured for social context understanding.

    Optimized for parsing who-said-what-to-whom in conversations.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_seq_len=256,  # Longer for conversations
            vocab_size=10000,
            num_classes=None,  # Embedding output
            checkpoint_path=checkpoint_path
        )
        print("[Social Context Facet] Initialized for conversation parsing")


class IntentDetectionFacet(TransformerFacet):
    """
    Transformer configured for intent classification.

    Classifies user intent from utterances.
    """

    INTENTS = [
        'greeting', 'farewell', 'question', 'command', 'statement',
        'emotional_expression', 'clarification', 'agreement', 'disagreement'
    ]

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(
            embed_dim=32,  # Smaller for classification
            num_heads=2,
            num_layers=1,
            ff_dim=128,
            max_seq_len=64,  # Single utterances
            vocab_size=10000,
            num_classes=len(self.INTENTS),
            checkpoint_path=checkpoint_path
        )
        print(f"[Intent Detection Facet] {len(self.INTENTS)} intent classes")

    async def detect_intent(self, text: str) -> Dict[str, float]:
        """Detect intent from text, returning probabilities for each class."""
        output = await self.process_text(text)
        if output.classification:
            return {
                self.INTENTS[int(k)]: v
                for k, v in output.classification.items()
            }
        return {}


class ReferenceResolutionFacet(TransformerFacet):
    """
    Transformer configured for pronoun/reference resolution.

    Resolves "it", "they", "that" to their referents.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            max_seq_len=128,
            vocab_size=10000,
            num_classes=None,  # Outputs attention for resolution
            checkpoint_path=checkpoint_path
        )
        print("[Reference Resolution Facet] Initialized for pronoun resolution")

    async def resolve_references(
        self,
        text: str,
        entities: List[str]
    ) -> Dict[str, str]:
        """
        Resolve pronouns to entities based on attention patterns.

        Args:
            text: Text containing pronouns
            entities: List of possible referents

        Returns:
            Dict mapping pronouns to most likely referents
        """
        # Simple word tokenization
        words = text.lower().split()
        pronouns = ['it', 'they', 'them', 'he', 'she', 'this', 'that']

        output = await self.process_text(text)

        resolutions = {}
        for i, word in enumerate(words):
            if word in pronouns and i < len(output.attention_weights):
                # Find which entity this pronoun attends to most
                attn = output.attention_weights[i]
                for j, w in enumerate(words):
                    if w in [e.lower() for e in entities]:
                        if j < len(attn):
                            entity_idx = [e.lower() for e in entities].index(w)
                            if word not in resolutions or attn[j] > resolutions[word][1]:
                                resolutions[word] = (entities[entity_idx], attn[j])

        return {k: v[0] for k, v in resolutions.items()}


if __name__ == "__main__":
    """Test Transformer Facet."""
    import asyncio

    async def test():
        # Test basic transformer
        print("\n=== Testing Transformer Facet ===")
        facet = TransformerFacet(
            embed_dim=32,
            num_heads=2,
            num_layers=1
        )

        output = await facet.process_text("The cat sat on the mat because it was tired")

        print(f"Context embedding dim: {len(output.context_embedding)}")
        print(f"Attention shape: {len(output.attention_weights)}x{len(output.attention_weights[0]) if output.attention_weights else 0}")
        print(f"Processing time: {output.processing_time_ms:.2f}ms")
        print(f"Top attended tokens: {output.top_attended_tokens}")

        # Test intent detection
        print("\n=== Testing Intent Detection ===")
        intent_facet = IntentDetectionFacet()
        intents = await intent_facet.detect_intent("Hello, how are you?")
        print(f"Detected intents: {intents}")

        # Test reference resolution
        print("\n=== Testing Reference Resolution ===")
        ref_facet = ReferenceResolutionFacet()
        refs = await ref_facet.resolve_references(
            "The cat saw the dog and it ran away",
            ["cat", "dog"]
        )
        print(f"Resolved references: {refs}")

        print("\n=== Execution Stats ===")
        print(facet.get_execution_stats())

    asyncio.run(test())
