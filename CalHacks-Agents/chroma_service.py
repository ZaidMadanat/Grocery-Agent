"""Chroma VectorDB service with dependency-light embeddings."""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Legacy ChromaDB telemetry variables kept for compatibility with existing
# deployment scripts. The in-memory fallback ignores them but we keep the
# environment toggles harmlessly set to avoid surprising users that may still
# rely on them elsewhere.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NOFILE"] = "1"


class _InMemoryCollection:
    """Mimics a subset of ChromaDB's collection API in memory."""

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self._entries: List[Dict[str, Any]] = []

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for item in zip(ids, embeddings, documents, metadatas):
            entry_id, embedding, document, metadata = item
            self._entries.append(
                {
                    "id": entry_id,
                    "embedding": embedding,
                    "document": document,
                    "metadata": metadata,
                }
            )

    def _filter(self, where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not where:
            return list(self._entries)

        def matches(entry: Dict[str, Any]) -> bool:
            for key, value in where.items():
                if entry["metadata"].get(key) != value:
                    return False
            return True

        return [entry for entry in self._entries if matches(entry)]

    def get(self, where: Optional[Dict[str, Any]] = None, limit: int = 0) -> Dict[str, Any]:
        results = self._filter(where)
        if limit:
            results = results[:limit]

        return {
            "ids": [entry["id"] for entry in results],
            "documents": [entry["document"] for entry in results],
            "metadatas": [entry["metadata"] for entry in results],
            "embeddings": [entry["embedding"] for entry in results],
        }

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        # Guard against division by zero
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(
        self,
        query_embeddings: List[List[float]],
        where: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
    ) -> Dict[str, Any]:
        filtered = self._filter(where)
        query_ids: List[List[str]] = []
        query_docs: List[List[str]] = []
        query_metas: List[List[Dict[str, Any]]] = []
        query_distances: List[List[float]] = []

        for query_embedding in query_embeddings:
            scored = [
                (
                    1.0 - self._cosine_similarity(query_embedding, entry["embedding"]),
                    entry,
                )
                for entry in filtered
            ]
            scored.sort(key=lambda item: item[0])  # lower distance is better
            top = scored[:n_results]

            query_ids.append([entry["id"] for _, entry in top])
            query_docs.append([entry["document"] for _, entry in top])
            query_metas.append([entry["metadata"] for _, entry in top])
            query_distances.append([distance for distance, _ in top])

        return {
            "ids": query_ids,
            "documents": query_docs,
            "metadatas": query_metas,
            "distances": query_distances,
        }


class _InMemoryChromaClient:
    def __init__(self) -> None:
        self._collections: Dict[str, _InMemoryCollection] = {}

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> _InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = _InMemoryCollection(name, metadata)
        return self._collections[name]


class _HashedEmbeddingModel:
    """Deterministic pseudo-embedding fallback.

    Avoids heavy ML dependencies (PyTorch, sentence-transformers) which are
    known to crash on some local Python 3.12 + Apple Silicon setups.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, text: str) -> List[float]:
        if text is None:
            text = ""

        seed = text.encode("utf-8", errors="ignore") or b"\x00"
        digest = hashlib.sha256(seed).digest()
        vector: List[float] = []

        while len(vector) < self.dimension:
            for byte in digest:
                vector.append((byte / 255.0) * 2.0 - 1.0)
                if len(vector) >= self.dimension:
                    break
            else:
                digest = hashlib.sha256(digest + seed).digest()

        return vector


class ChromaService:
    def __init__(self):
        self.client = _InMemoryChromaClient()
        
        # Initialize embedding model (hashed fallback to avoid heavy deps)
        dim = int(os.getenv("CHROMA_EMBED_DIM", "384"))
        self.embedding_model = _HashedEmbeddingModel(dimension=dim)
        logging.getLogger(__name__).info(
            "ChromaService using hashed embedding fallback (dimension=%s)", dim
        )
        
        # Collections for different data types
        self.recipe_collection = self.client.get_or_create_collection(
            name="recipes",
            metadata={"description": "Recipe embeddings for similarity search"}
        )

        self.preference_collection = self.client.get_or_create_collection(
            name="user_preferences",
            metadata={"description": "User preference embeddings"}
        )

        self.user_context_collection = self.client.get_or_create_collection(
            name="user_context",
            metadata={"description": "User dietary context and history"}
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured embedding model."""
        return list(self.embedding_model.encode(text))
    
    def store_recipe(self, recipe_data: Dict[str, Any], embedding: List[float]) -> str:
        """Store recipe entry with embedding data."""
        recipe_id = str(uuid.uuid4())
        
        self.recipe_collection.add(
            ids=[recipe_id],
            embeddings=[embedding],
            documents=[json.dumps(recipe_data)],
            metadatas=[{
                "user_id": recipe_data["user_id"],
                "meal_type": recipe_data["meal_type"],
                "cuisine": recipe_data.get("cuisine", ""),
                "calories": recipe_data.get("calories", 0),
                "created_at": datetime.utcnow().isoformat()
            }]
        )
        
        return recipe_id
    
    def store_user_preference(self, user_id: int, preference_data: Dict[str, Any], embedding: List[float]) -> str:
        """Store user preference entry with embedding data."""
        pref_id = str(uuid.uuid4())
        
        self.preference_collection.add(
            ids=[pref_id],
            embeddings=[embedding],
            documents=[json.dumps(preference_data)],
            metadatas=[{
                "user_id": user_id,
                "preference_type": preference_data["preference_type"],
                "item_type": preference_data["item_type"],
                "strength": preference_data.get("strength", 1.0),
                "created_at": datetime.utcnow().isoformat()
            }]
        )
        
        return pref_id
    
    def get_user_preferences(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's preference history"""
        results = self.preference_collection.get(
            where={"user_id": user_id},
            limit=limit
        )
        
        preferences = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                preferences.append({
                    "id": results["ids"][i],
                    "data": json.loads(doc),
                    "metadata": results["metadatas"][i]
                })
        
        return preferences
    
    def find_similar_recipes(self, user_id: int, query_embedding: List[float], 
                           meal_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find recipes similar to user preferences"""
        results = self.recipe_collection.query(
            query_embeddings=[query_embedding],
            where={"user_id": user_id, "meal_type": meal_type},
            n_results=limit
        )
        
        recipes = []
        for i, doc in enumerate(results["documents"][0]):
            recipes.append({
                "id": results["ids"][0][i],
                "data": json.loads(doc),
                "metadata": results["metadatas"][0][i],
                "similarity": results["distances"][0][i]
            })
        
        return recipes
    
    def get_user_dislikes(self, user_id: int) -> List[str]:
        """Get list of items user dislikes"""
        # Get all preferences for user, then filter for dislikes
        results = self.preference_collection.get(
            where={"user_id": user_id},
            limit=100
        )
        
        dislikes = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                data = json.loads(doc)
                if data.get("preference_type") == "disliked":
                    dislikes.append(data["item_name"])
        
        return dislikes
    
    def build_preference_context(self, preferences: List[Dict[str, Any]], dislikes: List[str]) -> str:
        """Build context string from user preferences"""
        context_parts = []
        
        # Add liked items
        liked_items = [p["data"]["item_name"] for p in preferences if p["data"]["preference_type"] == "liked"]
        if liked_items:
            context_parts.append(f"Likes: {', '.join(liked_items)}")
        
        # Add disliked items
        if dislikes:
            context_parts.append(f"Dislikes: {', '.join(dislikes)}")
        
        # Add context from preferences
        contexts = [p["data"].get("context", "") for p in preferences if p["data"].get("context")]
        if contexts:
            context_parts.append(f"Context: {'; '.join(contexts)}")
        
        return "; ".join(context_parts) if context_parts else "No specific preferences recorded"
