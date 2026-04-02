# interface
from typing import Protocol, List, Sequence, Tuple
import math


# =========================================================
#  Exceptions
# =========================================================
class RagError(Exception):
    """General error in Rag module"""
    pass
class EmbeddingDimensionError(RagError):
    """Embedding vectors have incompatible dimensions"""
    pass
class EmbeddingModelError(RagError):
    """Embedding model failed to produce a valid embedding"""
    pass
class InvalidKError(RagError):
    """Invalid k parameter (k <= 0 or k > number of candidates)"""
    pass
class LowConfidenceMatchError(RagError):
    """No candidate is similar enough (confidence threshold not met)"""
    pass


# =========================================================
#  Embedding model interface
# =========================================================

class EmbeddingModel(Protocol):

    def embed(self, text: str) -> Sequence[float]:
        """
        Convert text into an embedding vector
        """


# =========================================================
#  Distance function
# =========================================================

def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """
    distance between a and b
    calculate cosine similarity (between -1 and 1)
    => cosine distance = 1- cosine similarity
    => the smaller, the more similar (between 0 and 1)
    """

    if len(a) != len(b):
        raise EmbeddingDimensionError(
            f"Vectors have different dimensions: {len(a)} vs {len(b)}"
        )

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot_product += x * y
        norm_a += x * x
        norm_b += y * y

    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise RagError("Cosine distance for zero vector is undefined")

    similarity = dot_product / (norm_a * norm_b)

    # numerical stabilisation (rounding error)
    if similarity > 1.0:
        similarity = 1.0
    elif similarity < -1.0:
        similarity = -1.0

    distance = 1.0 - similarity
    return distance


# =========================================================
#  RAG (with confidence threshold)
# =========================================================

def RAG(
    query: str,
    emb_column: List[Sequence[float]],
    k: int,
    emb_model: EmbeddingModel,
    *,
    confidence_threshold: float = 0.4
) -> List[int]:

    """
    In-memory semantic nearest-neighbor search used for SQL repair.

    IMPORTANT:
    ---------
    This function is NOT used for knowledge retrieval.
    It is used exclusively to repair invalid SQL identifiers
    (e.g. table names or column names) based on semantic similarity.

    Workflow:
    ---------
    1. The invalid identifier (query) is embedded using the embedding model.
    2. Each candidate identifier embedding is compared via cosine distance.
    3. Candidates are sorted by increasing distance.
    4. The closest match is accepted ONLY if its distance is below
       the confidence threshold.
    5. The indices of the top-k candidates are returned.

    Parameters:
    ----------
    query : str
        Invalid identifier extracted from a database error message
        (e.g. "identifier", "cities", "zip").
    emb_column : List[Sequence[float]]
        Embeddings of valid candidate identifiers from the database schema
        (table names or column names).
    k : int
        Number of nearest neighbors to return (usually k=1).
    emb_model : EmbeddingModel
        Local embedding model (e.g. all-MiniLM-L6-v2 or all-mpnet-base-v2).
    confidence_threshold : float
        Maximum allowed cosine distance for accepting a repair.
        Larger distances indicate low semantic similarity.

    Returns:
    --------
    List[int]
        Indices of the top-k most similar candidates.

    Raises:
    -------
    RagError
        If emb_column is empty or another general error occurs.
    InvalidKError
        If k is invalid (e.g. k <= 0 or k > number of documents).
    EmbeddingModelError
        If something goes wrong when creating the question embedding.
    EmbeddingDimensionError
        If the dimensions of the question embedding and document embeddings do not match.
    LowConfidenceMatchError
        If no candidate is sufficiently similar.
    """

    # -----------------------------
    # Basic checks
    # -----------------------------
    if not emb_column:
        raise RagError("emb_column is empty - no candidates to compare")
    if k <= 0:
        raise InvalidKError("k must be greater than 0")
    if k > len(emb_column):
        raise InvalidKError(
            f"k = {k} is greater than the number of candidates = {len(emb_column)}"
        )

    # -----------------------------
    # Query embedding
    # -----------------------------
    try:
        query_vec = emb_model.embed(query)
    except Exception as e:
        raise EmbeddingModelError(f"Error calculating query embedding: {e}")

    if query_vec is None or len(query_vec) == 0:
        raise EmbeddingModelError("Embedding model returned an empty embedding")

    emb_dim = len(emb_column[0])
    if len(query_vec) != emb_dim:
        raise EmbeddingDimensionError(
            f"Embedding dimensions do not match: "
            f"query has {len(query_vec)}, candidates have {emb_dim}"
        )

    # -----------------------------
    # Distance calculation
    # -----------------------------
    dists: List[Tuple[int, float]] = []

    for idx, cand_vec in enumerate(emb_column):
        if len(cand_vec) != emb_dim:
            raise EmbeddingDimensionError(
                f"Candidate {idx} has embedding dimension {len(cand_vec)}, "
                f"expected {emb_dim}"
            )

        dist = cosine_distance(query_vec, cand_vec)
        dists.append((idx, dist))

    # -----------------------------
    # argmin(distance)
    # -----------------------------
    dists.sort(key=lambda x: x[1])
    best_idx, best_dist = dists[0]

    # -----------------------------
    # Confidence check
    # -----------------------------
    if best_dist > confidence_threshold:
        raise LowConfidenceMatchError(
            f"Best distance {best_dist:.3f} exceeds threshold {confidence_threshold}"
        )

    # -----------------------------
    # Return top-k indices
    # -----------------------------
    return [idx for idx, _ in dists[:k]]