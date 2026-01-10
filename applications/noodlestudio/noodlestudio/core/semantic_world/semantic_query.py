# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Semantic Query System - Click-to-inspect with CLIP embeddings
#
#   "What did you click on? The universe will tell you." Prov...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.semantic_world.semantic_query
# PURPOSE:  Semantic Query
# LAYER:    Studio / Semantic World
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SplatHitInfo, SemanticSearchResult, SemanticMatch, CLIPEmbeddingGenerator, CLIPEmbeddingIndex
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class SplatHitInfo:
    """Result of clicking on a Gaussian splat."""

    # Entity identification
    entity_id: str = ""
    entity_type: str = ""          # "noodling", "prim", "environment"
    display_name: str = ""

    # Body part (for noodlings)
    body_part: str = ""            # "left_knee", "head", "torso"
    body_region: str = ""          # "lower_body", "upper_body", "head"

    # Skeletal binding
    primary_bone: str = ""         # "leftLowerLeg"
    bone_indices: List[int] = field(default_factory=list)
    bone_weights: List[float] = field(default_factory=list)

    # Position
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gaussian_index: int = -1

    # Semantic info
    semantic_label: str = ""
    confidence: float = 0.0

    # For UI display
    def summary(self) -> str:
        """Human-readable summary of the hit."""
        if not self.entity_id:
            return "Nothing selected"

        parts = [f"Entity: {self.display_name or self.entity_id}"]

        if self.body_part:
            parts.append(f"Body part: {self.body_part}")
        if self.primary_bone:
            parts.append(f"Bone: {self.primary_bone}")
        if self.semantic_label:
            parts.append(f"Label: {self.semantic_label}")

        parts.append(f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})")

        return " | ".join(parts)


@dataclass
class SemanticSearchResult:
    """Result of a semantic/CLIP search."""

    query: str
    matches: List['SemanticMatch'] = field(default_factory=list)
    search_time_ms: float = 0.0


@dataclass
class SemanticMatch:
    """Single match from semantic search."""

    entity_id: str
    body_part: str
    similarity: float              # 0-1, higher = better match
    position: Tuple[float, float, float]
    gaussian_indices: List[int]    # Matching Gaussians


# =============================================================================
# Ray-Gaussian Intersection
# =============================================================================

def ray_gaussian_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    gaussian_pos: np.ndarray,
    gaussian_scale: np.ndarray,
    gaussian_rotation: np.ndarray,
    threshold: float = 2.0
) -> Tuple[bool, float]:
    """
    Test if ray intersects a Gaussian splat.

    Uses Mahalanobis distance from ray to Gaussian center.
    Gaussians are treated as ellipsoids defined by their covariance.

    Args:
        ray_origin: Ray start point (3,)
        ray_direction: Ray direction (normalized) (3,)
        gaussian_pos: Gaussian center (3,)
        gaussian_scale: Gaussian scale (3,)
        gaussian_rotation: Gaussian rotation quaternion (4,)
        threshold: Mahalanobis distance threshold (default 2.0 = ~95% of probability mass)

    Returns:
        (hit, distance) where hit=True if intersection, distance is to center
    """
    # Compute closest point on ray to Gaussian center
    to_gaussian = gaussian_pos - ray_origin
    t = np.dot(to_gaussian, ray_direction)

    if t < 0:
        # Gaussian is behind ray origin
        return False, float('inf')

    closest_point = ray_origin + t * ray_direction
    diff = closest_point - gaussian_pos

    # Build covariance matrix from scale and rotation
    cov = _build_covariance(gaussian_scale, gaussian_rotation)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return False, float('inf')

    # Mahalanobis distance
    mahal_dist = np.sqrt(diff @ cov_inv @ diff)

    if mahal_dist < threshold:
        return True, t

    return False, float('inf')


def _build_covariance(scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build 3x3 covariance matrix from scale and rotation quaternion."""
    # Scale matrix (diagonal)
    S = np.diag(scale.astype(np.float64) ** 2)

    # Rotation matrix from quaternion (w, x, y, z)
    w, x, y, z = rotation
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)

    # Covariance = R @ S @ R.T
    return R @ S @ R.T


# =============================================================================
# CLIP Embedding Search
# =============================================================================

# =============================================================================
# CLIP Embedding Generator
# =============================================================================

class CLIPEmbeddingGenerator:
    """
    Generates CLIP embeddings for RadianceAsset semantic labels.

    Two modes:
    1. Text-based: Encode semantic labels ("left_hand" -> embedding)
    2. Visual (future): Render views and encode images

    Text-based is fast (~50ms) and handles body part queries well.
    """

    def __init__(self, device: str = None):
        """
        Initialize generator.

        Args:
            device: 'mps', 'cuda', 'cpu', or None for auto-detect
        """
        self._model = None
        self._tokenizer = None
        self._device = device
        self._embed_dim = 512  # CLIP ViT-B/32 dimension
        self._label_cache: Dict[str, np.ndarray] = {}

    def _ensure_model(self):
        """Lazy load CLIP model."""
        if self._model is not None:
            return True

        import torch

        # Auto-detect device
        if self._device is None:
            if torch.backends.mps.is_available():
                self._device = 'mps'
            elif torch.cuda.is_available():
                self._device = 'cuda'
            else:
                self._device = 'cpu'

        # Try transformers first (better download handling with progress)
        if self._try_load_transformers(torch):
            return True

        # Fall back to open_clip
        if self._try_load_open_clip(torch):
            return True

        logger.error("No CLIP model available - install transformers or open_clip")
        return False

    def _try_load_transformers(self, torch) -> bool:
        """Try loading CLIP via transformers library."""
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info("Loading CLIP model via transformers...")
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model = self._model.to(self._device)
            self._model.eval()
            self._backend = 'transformers'

            logger.info(f"CLIP model loaded via transformers on {self._device}")
            return True

        except Exception as e:
            logger.debug(f"transformers CLIP failed: {e}")
            return False

    def _try_load_open_clip(self, torch) -> bool:
        """Try loading CLIP via open_clip library."""
        try:
            import open_clip

            logger.info("Loading CLIP model via open_clip...")
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self._tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self._model = self._model.to(self._device)
            self._model.eval()
            self._backend = 'open_clip'

            logger.info(f"CLIP model loaded via open_clip on {self._device}")
            return True

        except Exception as e:
            logger.debug(f"open_clip CLIP failed: {e}")
            return False

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text to CLIP embedding.

        Args:
            text: Text to encode

        Returns:
            (512,) numpy array, or None on failure
        """
        if not self._ensure_model():
            return None

        # Check cache
        if text in self._label_cache:
            return self._label_cache[text]

        try:
            import torch

            if getattr(self, '_backend', 'open_clip') == 'transformers':
                # Transformers API
                inputs = self._tokenizer(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    embedding = self._model.get_text_features(**inputs)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            else:
                # Open_clip API
                tokens = self._tokenizer([text]).to(self._device)
                with torch.no_grad():
                    embedding = self._model.encode_text(tokens)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            result = embedding.cpu().numpy().flatten()
            self._label_cache[text] = result
            return result

        except Exception as e:
            logger.error(f"CLIP text encoding failed: {e}")
            return None

    def encode_texts_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Encode multiple texts in a batch.

        Args:
            texts: List of texts to encode

        Returns:
            (N, 512) numpy array, or None on failure
        """
        if not self._ensure_model():
            return None

        if not texts:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        try:
            import torch

            if getattr(self, '_backend', 'open_clip') == 'transformers':
                # Transformers API
                inputs = self._tokenizer(text=texts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    embeddings = self._model.get_text_features(**inputs)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            else:
                # Open_clip API
                tokens = self._tokenizer(texts).to(self._device)
                with torch.no_grad():
                    embeddings = self._model.encode_text(tokens)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"CLIP batch encoding failed: {e}")
            return None

    def generate_embeddings_for_asset(
        self,
        asset: 'RadianceAsset',
        label_format: str = "natural"
    ) -> Optional[np.ndarray]:
        """
        Generate CLIP embeddings for all Gaussians in an asset.

        Uses semantic_labels to create text embeddings. Each unique label
        is encoded once, then assigned to all Gaussians with that label.

        Args:
            asset: RadianceAsset with semantic_labels populated
            label_format: How to format labels for CLIP
                - "natural": "left hand", "head", "right lower leg"
                - "descriptive": "the left hand of a character"
                - "raw": use labels as-is

        Returns:
            (N, 512) numpy array of embeddings, or None on failure
        """
        if not asset.semantic_labels:
            logger.warning("Asset has no semantic labels - generating uniform embeddings")
            # Generate uniform embeddings (all same)
            n = asset.gaussian_count
            default_emb = self.encode_text("unknown body part")
            if default_emb is None:
                return None
            return np.tile(default_emb, (n, 1))

        # Get unique labels
        unique_labels = list(set(asset.semantic_labels))
        unique_labels = [l for l in unique_labels if l]  # Filter empty

        if not unique_labels:
            logger.warning("No valid semantic labels in asset")
            return None

        # Format labels for CLIP
        formatted_labels = []
        for label in unique_labels:
            if label_format == "natural":
                # Convert "leftHand" -> "left hand"
                formatted = self._format_label_natural(label)
            elif label_format == "descriptive":
                formatted = f"the {self._format_label_natural(label)} of a character"
            else:
                formatted = label
            formatted_labels.append(formatted)

        # Encode all unique labels
        label_embeddings = self.encode_texts_batch(formatted_labels)
        if label_embeddings is None:
            return None

        # Build label -> embedding map
        label_to_embedding = {
            unique_labels[i]: label_embeddings[i]
            for i in range(len(unique_labels))
        }

        # Assign embeddings to all Gaussians
        n = asset.gaussian_count
        embeddings = np.zeros((n, self._embed_dim), dtype=np.float32)

        # Default embedding for unlabeled Gaussians
        default_emb = self.encode_text("body part")
        if default_emb is None:
            default_emb = np.zeros(self._embed_dim, dtype=np.float32)

        for i in range(n):
            label = asset.semantic_labels[i] if i < len(asset.semantic_labels) else ""
            if label and label in label_to_embedding:
                embeddings[i] = label_to_embedding[label]
            else:
                embeddings[i] = default_emb

        logger.info(f"Generated CLIP embeddings: {n} Gaussians, {len(unique_labels)} unique labels")
        return embeddings

    def _format_label_natural(self, label: str) -> str:
        """
        Format a label for natural language.

        "leftHand" -> "left hand"
        "left_hand" -> "left hand"
        "rightLowerLeg" -> "right lower leg"
        """
        import re

        # Handle camelCase
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)

        # Handle snake_case
        result = result.replace('_', ' ')

        return result.lower().strip()

    @property
    def embed_dim(self) -> int:
        """Get embedding dimension."""
        return self._embed_dim


def populate_asset_embeddings(
    asset: 'RadianceAsset',
    generator: Optional[CLIPEmbeddingGenerator] = None
) -> bool:
    """
    Populate CLIP embeddings for a RadianceAsset.

    Convenience function that generates embeddings from semantic labels
    and assigns them to the asset.

    Args:
        asset: RadianceAsset to populate
        generator: CLIPEmbeddingGenerator (creates one if None)

    Returns:
        True if successful, False otherwise
    """
    if generator is None:
        generator = CLIPEmbeddingGenerator()

    embeddings = generator.generate_embeddings_for_asset(asset)

    if embeddings is not None:
        asset.clip_embeddings = embeddings
        logger.info(f"Populated {asset.gaussian_count} CLIP embeddings for asset")
        return True

    return False


class CLIPEmbeddingIndex:
    """
    Index for fast CLIP embedding search.

    Enables natural language queries like:
    - "Red's left hand"
    - "the chair in the corner"
    - "something red"
    """

    def __init__(self):
        # entity_id -> {gaussian_idx -> embedding}
        self._embeddings: Dict[str, Dict[int, np.ndarray]] = {}

        # entity_id -> {gaussian_idx -> metadata}
        self._metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # Precomputed for fast search
        self._all_embeddings: Optional[np.ndarray] = None
        self._embedding_map: List[Tuple[str, int]] = []  # (entity_id, gaussian_idx)

        self._dirty = True

    def add_entity(
        self,
        entity_id: str,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ):
        """
        Add entity's CLIP embeddings to index.

        Args:
            entity_id: Entity identifier
            embeddings: (N, embed_dim) array of CLIP embeddings
            metadata: Per-Gaussian metadata (body_part, position, etc.)
        """
        self._embeddings[entity_id] = {}
        self._metadata[entity_id] = {}

        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            self._embeddings[entity_id][i] = emb
            self._metadata[entity_id][i] = meta

        self._dirty = True
        logger.info(f"Added {len(embeddings)} embeddings for entity: {entity_id}")

    def remove_entity(self, entity_id: str):
        """Remove entity from index."""
        self._embeddings.pop(entity_id, None)
        self._metadata.pop(entity_id, None)
        self._dirty = True

    def _rebuild_index(self):
        """Rebuild the flat search index."""
        if not self._dirty:
            return

        all_embs = []
        self._embedding_map = []

        for entity_id, emb_dict in self._embeddings.items():
            for gaussian_idx, emb in emb_dict.items():
                all_embs.append(emb)
                self._embedding_map.append((entity_id, gaussian_idx))

        if all_embs:
            self._all_embeddings = np.stack(all_embs)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self._all_embeddings, axis=1, keepdims=True)
            self._all_embeddings = self._all_embeddings / (norms + 1e-8)
        else:
            self._all_embeddings = None

        self._dirty = False
        logger.info(f"Rebuilt CLIP index with {len(self._embedding_map)} embeddings")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        entity_filter: Optional[str] = None
    ) -> List[SemanticMatch]:
        """
        Search for Gaussians matching a query embedding.

        Args:
            query_embedding: (embed_dim,) CLIP embedding of query
            top_k: Number of results to return
            entity_filter: Only search within this entity (optional)

        Returns:
            List of SemanticMatch results sorted by similarity
        """
        self._rebuild_index()

        if self._all_embeddings is None or len(self._all_embeddings) == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Cosine similarity (dot product of normalized vectors)
        similarities = self._all_embeddings @ query_norm

        # Get top k indices
        if entity_filter:
            # Filter to only this entity
            valid_mask = np.array([
                entity_id == entity_filter
                for entity_id, _ in self._embedding_map
            ])
            similarities = np.where(valid_mask, similarities, -np.inf)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:
                continue

            entity_id, gaussian_idx = self._embedding_map[idx]
            meta = self._metadata.get(entity_id, {}).get(gaussian_idx, {})

            results.append(SemanticMatch(
                entity_id=entity_id,
                body_part=meta.get('body_part', ''),
                similarity=float(similarities[idx]),
                position=tuple(meta.get('position', (0, 0, 0))),
                gaussian_indices=[gaussian_idx]
            ))

        return results

    def search_text(
        self,
        query_text: str,
        clip_model: Any,
        top_k: int = 10
    ) -> List[SemanticMatch]:
        """
        Search using natural language query.

        Args:
            query_text: Natural language query ("Red's left hand")
            clip_model: CLIP model with encode_text method
            top_k: Number of results

        Returns:
            Matching Gaussians
        """
        # Encode query text
        try:
            query_embedding = clip_model.encode_text(query_text)
            return self.search(query_embedding, top_k=top_k)
        except Exception as e:
            logger.error(f"CLIP text encoding failed: {e}")
            return []


# =============================================================================
# Semantic Query Engine
# =============================================================================

class SemanticQueryEngine:
    """
    Main engine for semantic queries on Gaussian scenes.

    Combines:
    - Ray casting for click-to-select
    - CLIP search for natural language queries
    - Entity registry for metadata lookup
    """

    def __init__(self, auto_generate_embeddings: bool = True):
        """
        Initialize query engine.

        Args:
            auto_generate_embeddings: Auto-generate CLIP embeddings for assets
                                      that have semantic labels but no embeddings
        """
        self.clip_index = CLIPEmbeddingIndex()

        # entity_id -> RadianceAsset
        self._assets: Dict[str, Any] = {}

        # entity_id -> display_name
        self._entity_names: Dict[str, str] = {}

        # entity_id -> entity_type
        self._entity_types: Dict[str, str] = {}

        # CLIP embedding generator
        self._generator: Optional[CLIPEmbeddingGenerator] = None
        self._auto_generate = auto_generate_embeddings

    def register_entity(
        self,
        entity_id: str,
        asset: 'RadianceAsset',
        display_name: str = "",
        entity_type: str = "noodling"
    ):
        """
        Register an entity's radiance asset for queries.

        Args:
            entity_id: Unique entity identifier
            asset: RadianceAsset with Gaussians and metadata
            display_name: Human-readable name
            entity_type: "noodling", "prim", "environment"
        """
        self._assets[entity_id] = asset
        self._entity_names[entity_id] = display_name or entity_id
        self._entity_types[entity_id] = entity_type

        # Auto-generate CLIP embeddings if needed
        if self._auto_generate and not asset.has_clip and asset.semantic_labels:
            logger.info(f"Auto-generating CLIP embeddings for {entity_id}")
            if self._generator is None:
                self._generator = CLIPEmbeddingGenerator()
            populate_asset_embeddings(asset, self._generator)

        # Index CLIP embeddings if available
        if hasattr(asset, 'clip_embeddings') and asset.clip_embeddings is not None:
            # Build metadata for each Gaussian
            metadata = []
            for i in range(asset.gaussian_count):
                # Get body part label
                body_part = ''
                if asset.semantic_labels and i < len(asset.semantic_labels):
                    body_part = asset.semantic_labels[i]

                # Get body region
                body_region = ''
                if asset.body_regions is not None and i < len(asset.body_regions):
                    body_region = asset.get_body_region(i)

                meta = {
                    'position': tuple(asset.positions[i]),
                    'body_part': body_part,
                    'body_region': body_region,
                }

                # Add bone binding info if available
                if asset.skin_bone_indices is not None and i < len(asset.skin_bone_indices):
                    meta['bone_indices'] = list(asset.skin_bone_indices[i])
                    meta['bone_weights'] = list(asset.skin_bone_weights[i])
                    if asset.skeleton and asset.skeleton.bones:
                        primary_idx = int(asset.skin_bone_indices[i][0])
                        if 0 <= primary_idx < len(asset.skeleton.bones):
                            meta['primary_bone'] = asset.skeleton.bones[primary_idx].name

                metadata.append(meta)

            self.clip_index.add_entity(entity_id, asset.clip_embeddings, metadata)

        logger.info(f"Registered entity for queries: {entity_id} ({entity_type})")

    def unregister_entity(self, entity_id: str):
        """Remove entity from query engine."""
        self._assets.pop(entity_id, None)
        self._entity_names.pop(entity_id, None)
        self._entity_types.pop(entity_id, None)
        self.clip_index.remove_entity(entity_id)

    def raycast(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        max_distance: float = 100.0
    ) -> Optional[SplatHitInfo]:
        """
        Find the Gaussian hit by a ray (click-to-select).

        Args:
            ray_origin: Ray start point in world space
            ray_direction: Ray direction (will be normalized)
            max_distance: Maximum ray distance

        Returns:
            SplatHitInfo if hit, None otherwise
        """
        ray_direction = ray_direction / (np.linalg.norm(ray_direction) + 1e-8)

        best_hit = None
        best_distance = max_distance

        for entity_id, asset in self._assets.items():
            for i in range(asset.gaussian_count):
                hit, dist = ray_gaussian_intersection(
                    ray_origin,
                    ray_direction,
                    asset.positions[i],
                    asset.scales[i],
                    asset.rotations[i]
                )

                if hit and dist < best_distance:
                    best_distance = dist

                    # Build hit info
                    hit_info = SplatHitInfo(
                        entity_id=entity_id,
                        entity_type=self._entity_types.get(entity_id, 'unknown'),
                        display_name=self._entity_names.get(entity_id, entity_id),
                        position=tuple(asset.positions[i]),
                        gaussian_index=i
                    )

                    # Add semantic info if available
                    if hasattr(asset, 'semantic_labels') and asset.semantic_labels:
                        hit_info.semantic_label = asset.semantic_labels[i]
                        hit_info.body_part = asset.semantic_labels[i]

                    if hasattr(asset, 'body_regions') and asset.body_regions:
                        hit_info.body_region = asset.body_regions[i]

                    # Add bone binding
                    if hasattr(asset, 'bone_indices') and asset.bone_indices is not None:
                        hit_info.bone_indices = list(asset.bone_indices[i])
                        hit_info.bone_weights = list(asset.bone_weights[i])

                        if hasattr(asset, 'skeleton') and asset.skeleton:
                            primary_idx = asset.bone_indices[i][0]
                            if primary_idx < len(asset.skeleton.bones):
                                hit_info.primary_bone = asset.skeleton.bones[primary_idx].name

                    best_hit = hit_info

        return best_hit

    def query_text(
        self,
        query: str,
        top_k: int = 5
    ) -> SemanticSearchResult:
        """
        Search using natural language query.

        Args:
            query: Natural language query ("Red's left hand", "something blue")
            top_k: Number of results

        Returns:
            SemanticSearchResult with matches
        """
        import time
        start = time.time()

        # Ensure generator is available
        if self._generator is None:
            self._generator = CLIPEmbeddingGenerator()

        # Encode query text
        query_embedding = self._generator.encode_text(query)

        if query_embedding is None:
            return SemanticSearchResult(query=query, matches=[], search_time_ms=0)

        # Search the index
        matches = self.clip_index.search(query_embedding, top_k=top_k)

        elapsed_ms = (time.time() - start) * 1000

        return SemanticSearchResult(
            query=query,
            matches=matches,
            search_time_ms=elapsed_ms
        )

    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get info about a registered entity."""
        if entity_id not in self._assets:
            return None

        asset = self._assets[entity_id]

        return {
            'entity_id': entity_id,
            'display_name': self._entity_names.get(entity_id, entity_id),
            'entity_type': self._entity_types.get(entity_id, 'unknown'),
            'gaussian_count': asset.gaussian_count,
            'has_skeleton': hasattr(asset, 'skeleton') and asset.skeleton is not None,
            'has_clip': hasattr(asset, 'clip_embeddings') and asset.clip_embeddings is not None,
            'bounds_min': tuple(asset.positions.min(axis=0)) if asset.gaussian_count > 0 else (0, 0, 0),
            'bounds_max': tuple(asset.positions.max(axis=0)) if asset.gaussian_count > 0 else (0, 0, 0),
        }

    def list_entities(self) -> List[str]:
        """Get list of registered entity IDs."""
        return list(self._assets.keys())


# =============================================================================
# Global Instance
# =============================================================================

_query_engine: Optional[SemanticQueryEngine] = None


def init_semantic_query_engine() -> SemanticQueryEngine:
    """Initialize the global semantic query engine."""
    global _query_engine
    _query_engine = SemanticQueryEngine()
    logger.info("Semantic query engine initialized")
    return _query_engine


def get_semantic_query_engine() -> Optional[SemanticQueryEngine]:
    """Get the global semantic query engine."""
    return _query_engine


# =============================================================================
# Convenience Functions
# =============================================================================

def click_to_inspect(
    ray_origin: Tuple[float, float, float],
    ray_direction: Tuple[float, float, float]
) -> Optional[SplatHitInfo]:
    """
    Convenience function for click-to-inspect.

    Args:
        ray_origin: Click ray origin in world space
        ray_direction: Click ray direction

    Returns:
        Hit info or None
    """
    engine = get_semantic_query_engine()
    if engine is None:
        return None

    return engine.raycast(
        np.array(ray_origin),
        np.array(ray_direction)
    )


def query_scene(query: str, top_k: int = 5) -> SemanticSearchResult:
    """
    Convenience function for natural language scene queries.

    Args:
        query: Natural language query
        top_k: Number of results

    Returns:
        Search results
    """
    engine = get_semantic_query_engine()
    if engine is None:
        return SemanticSearchResult(query=query)

    return engine.query_text(query, top_k=top_k)


__all__ = [
    # Data types
    'SplatHitInfo',
    'SemanticSearchResult',
    'SemanticMatch',

    # CLIP embedding generation
    'CLIPEmbeddingGenerator',
    'populate_asset_embeddings',

    # Search index
    'CLIPEmbeddingIndex',

    # Main engine
    'SemanticQueryEngine',

    # Module interface
    'init_semantic_query_engine',
    'get_semantic_query_engine',
    'click_to_inspect',
    'query_scene',

    # Ray intersection
    'ray_gaussian_intersection',
]

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
