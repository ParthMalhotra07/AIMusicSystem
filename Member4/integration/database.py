"""
Song Database Module

SQLite-based database for storing songs, features, and embeddings.
Provides persistent storage for user-uploaded songs.
"""

import sqlite3
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import numpy as np
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "songs.db"
DEFAULT_AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio_uploads"


class SongDatabase:
    """
    SQLite database for managing songs, features, and embeddings.
    
    Schema:
    - songs: song_id, filename, filepath, upload_date, duration, tempo, key
    - features: song_id, feature_vector (blob)
    - embeddings: song_id, embedding_vector (blob)
    """
    
    def __init__(
        self,
        db_path: str = None,
        audio_dir: str = None
    ):
        """
        Initialize database connection.
        
        Parameters
        ----------
        db_path : str, optional
            Path to SQLite database file
        audio_dir : str, optional
            Directory to store uploaded audio files
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.audio_dir = Path(audio_dir) if audio_dir else DEFAULT_AUDIO_DIR
        
        # Create directories if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _init_db(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Songs table - main song information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                song_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT,
                upload_date TEXT NOT NULL,
                duration_sec REAL,
                tempo REAL,
                key_name TEXT,
                key_confidence REAL,
                spectral_centroid REAL,
                rms REAL,
                is_user_uploaded INTEGER DEFAULT 1
            )
        ''')
        
        # Features table - high-dimensional feature vectors
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                song_id TEXT PRIMARY KEY,
                feature_vector BLOB NOT NULL,
                feature_dim INTEGER NOT NULL,
                FOREIGN KEY (song_id) REFERENCES songs(song_id)
            )
        ''')
        
        # Embeddings table - low-dimensional embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                song_id TEXT PRIMARY KEY,
                embedding_vector BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                FOREIGN KEY (song_id) REFERENCES songs(song_id)
            )
        ''')
        
        # MFCC features (for display/explainability)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mfcc_features (
                song_id TEXT PRIMARY KEY,
                mfcc_values TEXT NOT NULL,
                FOREIGN KEY (song_id) REFERENCES songs(song_id)
            )
        ''')
        
        # Chroma features (for display/explainability)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chroma_features (
                song_id TEXT PRIMARY KEY,
                chroma_values TEXT NOT NULL,
                FOREIGN KEY (song_id) REFERENCES songs(song_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_song_id(self, filename: str) -> str:
        """Generate a unique song ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Clean filename for ID
        clean_name = Path(filename).stem.replace(" ", "_").replace("-", "_")[:20]
        return f"song_{clean_name}_{timestamp}"
    
    def add_song(
        self,
        filename: str,
        features: np.ndarray,
        embedding: np.ndarray,
        metadata: Dict,
        audio_data: bytes = None,
        song_id: str = None
    ) -> str:
        """
        Add a new song to the database.
        
        Parameters
        ----------
        filename : str
            Original filename
        features : np.ndarray
            Feature vector from Member1
        embedding : np.ndarray
            Embedding vector from Member2
        metadata : dict
            Audio metadata (tempo, key, duration, etc.)
        audio_data : bytes, optional
            Raw audio file data to store
        song_id : str, optional
            Custom song ID (auto-generated if not provided)
            
        Returns
        -------
        song_id : str
            The assigned song ID
        """
        if song_id is None:
            song_id = self._generate_song_id(filename)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Store audio file if provided
            filepath = None
            if audio_data is not None:
                ext = Path(filename).suffix or '.mp3'
                filepath = str(self.audio_dir / f"{song_id}{ext}")
                with open(filepath, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"Audio saved to {filepath}")
            
            # Insert song record
            cursor.execute('''
                INSERT OR REPLACE INTO songs 
                (song_id, filename, filepath, upload_date, duration_sec, tempo, 
                 key_name, key_confidence, spectral_centroid, rms, is_user_uploaded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                song_id,
                filename,
                filepath,
                datetime.now().isoformat(),
                metadata.get('duration_seconds'),
                metadata.get('tempo_bpm'),
                metadata.get('key'),
                metadata.get('key_confidence'),
                metadata.get('spectral_centroid'),
                metadata.get('rms'),
                1  # is_user_uploaded
            ))
            
            # Insert features
            cursor.execute('''
                INSERT OR REPLACE INTO features (song_id, feature_vector, feature_dim)
                VALUES (?, ?, ?)
            ''', (
                song_id,
                features.astype(np.float32).tobytes(),
                len(features)
            ))
            
            # Insert embedding
            cursor.execute('''
                INSERT OR REPLACE INTO embeddings (song_id, embedding_vector, embedding_dim)
                VALUES (?, ?, ?)
            ''', (
                song_id,
                embedding.astype(np.float32).tobytes(),
                len(embedding)
            ))
            
            # Extract and store MFCC values from features for explainability
            # Fast mode features: [0:13] MFCC means
            if len(features) >= 13:
                mfcc_values = features[0:13].tolist()
                cursor.execute('''
                    INSERT OR REPLACE INTO mfcc_features (song_id, mfcc_values)
                    VALUES (?, ?)
                ''', (song_id, json.dumps(mfcc_values)))
            
            # Extract and store chroma values from features for explainability
            # Fast mode features: [35:47] chroma means
            if len(features) >= 47:
                chroma_values = features[35:47].tolist()
                cursor.execute('''
                    INSERT OR REPLACE INTO chroma_features (song_id, chroma_values)
                    VALUES (?, ?)
                ''', (song_id, json.dumps(chroma_values)))
            
            conn.commit()
            logger.info(f"Song '{filename}' added with ID: {song_id}")
            return song_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add song: {e}")
            raise
        finally:
            conn.close()
    
    def get_song(self, song_id: str) -> Optional[Dict]:
        """Get song information by ID."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.*, f.feature_vector, f.feature_dim, e.embedding_vector, e.embedding_dim
            FROM songs s
            LEFT JOIN features f ON s.song_id = f.song_id
            LEFT JOIN embeddings e ON s.song_id = e.song_id
            WHERE s.song_id = ?
        ''', (song_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'song_id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'upload_date': row[3],
            'duration_sec': row[4],
            'tempo': row[5],
            'key_name': row[6],
            'key_confidence': row[7],
            'spectral_centroid': row[8],
            'rms': row[9],
            'is_user_uploaded': bool(row[10]),
            'features': np.frombuffer(row[11], dtype=np.float32) if row[11] else None,
            'feature_dim': row[12],
            'embedding': np.frombuffer(row[13], dtype=np.float32) if row[13] else None,
            'embedding_dim': row[14]
        }
    
    def get_all_songs(self) -> List[Dict]:
        """Get all songs in the database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT song_id, filename, filepath, upload_date, duration_sec, 
                   tempo, key_name, spectral_centroid, rms, is_user_uploaded
            FROM songs
            ORDER BY upload_date DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'song_id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'upload_date': row[3],
            'duration_sec': row[4],
            'tempo': row[5],
            'key_name': row[6],
            'spectral_centroid': row[7],
            'rms': row[8],
            'is_user_uploaded': bool(row[9])
        } for row in rows]
    
    def get_all_embeddings(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all embeddings as numpy arrays.
        
        Parameters
        ----------
        normalize : bool
            If True, normalize embeddings to unit vectors for consistent
            cosine similarity comparisons. Default True.
        
        Returns
        -------
        embeddings : np.ndarray
            Matrix of all embeddings (N x embedding_dim)
        song_ids : np.ndarray
            Array of corresponding song IDs
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT song_id, embedding_vector, embedding_dim
            FROM embeddings
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return np.array([]), np.array([])
        
        song_ids = np.array([row[0] for row in rows])
        
        # Handle variable dimension embeddings
        raw_embeddings = [
            np.frombuffer(row[1], dtype=np.float32) 
            for row in rows
        ]
        
        # Find max dimension and pad shorter ones
        max_dim = max(len(e) for e in raw_embeddings)
        aligned_embeddings = []
        for emb in raw_embeddings:
            if len(emb) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(emb)] = emb
                aligned_embeddings.append(padded)
            else:
                aligned_embeddings.append(emb[:max_dim])
        
        embeddings = np.array(aligned_embeddings)
        
        # Normalize to unit vectors for consistent cosine similarity
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1e-8)  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings, song_ids
    
    def get_all_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all features as numpy arrays.
        
        Returns
        -------
        features : np.ndarray
            Matrix of all features (N x feature_dim)
        song_ids : np.ndarray
            Array of corresponding song IDs
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT song_id, feature_vector, feature_dim
            FROM features
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return np.array([]), np.array([])
        
        song_ids = np.array([row[0] for row in rows])
        
        # Handle variable dimension features
        raw_features = [
            np.frombuffer(row[1], dtype=np.float32) 
            for row in rows
        ]
        
        # Find max dimension and pad shorter ones
        max_dim = max(len(f) for f in raw_features)
        aligned_features = []
        for feat in raw_features:
            if len(feat) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(feat)] = feat
                aligned_features.append(padded)
            else:
                aligned_features.append(feat[:max_dim])
        
        features = np.array(aligned_features)
        
        return features, song_ids
    
    def delete_song(self, song_id: str) -> bool:
        """Delete a song from the database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Get filepath to delete audio file
            cursor.execute('SELECT filepath FROM songs WHERE song_id = ?', (song_id,))
            row = cursor.fetchone()
            
            if row and row[0] and os.path.exists(row[0]):
                os.remove(row[0])
            
            # Delete from all tables
            cursor.execute('DELETE FROM embeddings WHERE song_id = ?', (song_id,))
            cursor.execute('DELETE FROM features WHERE song_id = ?', (song_id,))
            cursor.execute('DELETE FROM mfcc_features WHERE song_id = ?', (song_id,))
            cursor.execute('DELETE FROM chroma_features WHERE song_id = ?', (song_id,))
            cursor.execute('DELETE FROM songs WHERE song_id = ?', (song_id,))
            
            conn.commit()
            logger.info(f"Song {song_id} deleted")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete song: {e}")
            return False
        finally:
            conn.close()
    
    def get_song_count(self) -> int:
        """Get total number of songs in database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM songs')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_mfcc_values(self, song_id: str) -> Optional[List[float]]:
        """Get MFCC values for a song."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT mfcc_values FROM mfcc_features WHERE song_id = ?', (song_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            return json.loads(row[0])
        return None
    
    def get_chroma_values(self, song_id: str) -> Optional[List[float]]:
        """Get chroma values for a song."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT chroma_values FROM chroma_features WHERE song_id = ?', (song_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            return json.loads(row[0])
        return None
    
    def get_all_songs_with_features(self) -> List[Dict]:
        """
        Get all songs with their MFCC and chroma features for explainability.
        
        Returns list of dicts with song info and features.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.song_id, s.filename, s.tempo, s.spectral_centroid, s.rms, s.duration_sec,
                   m.mfcc_values, c.chroma_values
            FROM songs s
            LEFT JOIN mfcc_features m ON s.song_id = m.song_id
            LEFT JOIN chroma_features c ON s.song_id = c.song_id
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            song_data = {
                'song_id': row[0],
                'filename': row[1],
                'tempo': row[2] or 0,
                'spectral_centroid': row[3] or 0,
                'rms': row[4] or 0,
                'duration_sec': row[5] or 0,
            }
            
            # Add MFCC columns
            if row[6]:
                mfcc_values = json.loads(row[6])
                for i, val in enumerate(mfcc_values):
                    song_data[f'mfcc_mean_{i+1}'] = val
            
            # Add chroma columns
            if row[7]:
                chroma_values = json.loads(row[7])
                for i, val in enumerate(chroma_values):
                    song_data[f'chroma_mean_{i+1}'] = val
            
            results.append(song_data)
        
        return results
    
    def import_from_numpy(
        self,
        features_path: str,
        embeddings_path: str,
        song_ids_path: str,
        features_csv_path: str = None
    ):
        """
        Import existing data from numpy files into database.
        
        Parameters
        ----------
        features_path : str
            Path to song_features.npy
        embeddings_path : str
            Path to song_embeddings.npy
        song_ids_path : str
            Path to song_ids.npy
        features_csv_path : str, optional
            Path to song_features.csv for additional metadata
        """
        import pandas as pd
        
        # Load numpy files
        features = np.load(features_path)
        embeddings = np.load(embeddings_path)
        song_ids = np.load(song_ids_path, allow_pickle=True)
        
        # Load CSV for metadata if available
        metadata_df = None
        if features_csv_path and os.path.exists(features_csv_path):
            metadata_df = pd.read_csv(features_csv_path)
            metadata_df.set_index('song_id', inplace=True)
        
        logger.info(f"Importing {len(song_ids)} songs from numpy files...")
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for i, song_id in enumerate(song_ids):
            song_id = str(song_id)
            
            # Get metadata from CSV if available
            meta = {}
            if metadata_df is not None and song_id in metadata_df.index:
                row = metadata_df.loc[song_id]
                meta = {
                    'duration_seconds': row.get('duration_sec'),
                    'tempo_bpm': row.get('tempo'),
                    'spectral_centroid': row.get('spectral_centroid'),
                    'rms': row.get('rms')
                }
            
            # Insert song
            cursor.execute('''
                INSERT OR REPLACE INTO songs 
                (song_id, filename, upload_date, duration_sec, tempo, spectral_centroid, rms, is_user_uploaded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                song_id,
                song_id,
                datetime.now().isoformat(),
                meta.get('duration_seconds'),
                meta.get('tempo_bpm'),
                meta.get('spectral_centroid'),
                meta.get('rms'),
                0  # Not user uploaded - imported
            ))
            
            # Insert features
            cursor.execute('''
                INSERT OR REPLACE INTO features (song_id, feature_vector, feature_dim)
                VALUES (?, ?, ?)
            ''', (song_id, features[i].astype(np.float32).tobytes(), features.shape[1]))
            
            # Insert embedding
            cursor.execute('''
                INSERT OR REPLACE INTO embeddings (song_id, embedding_vector, embedding_dim)
                VALUES (?, ?, ?)
            ''', (song_id, embeddings[i].astype(np.float32).tobytes(), embeddings.shape[1]))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Imported {len(song_ids)} songs into database")
    
    def export_to_numpy(self, output_dir: str):
        """
        Export database to numpy files (for compatibility).
        
        Parameters
        ----------
        output_dir : str
            Directory to save numpy files
        """
        import pandas as pd
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all data
        features, feature_ids = self.get_all_features()
        embeddings, embedding_ids = self.get_all_embeddings()
        songs = self.get_all_songs()
        
        # Save numpy files
        np.save(output_dir / "song_features.npy", features)
        np.save(output_dir / "song_embeddings.npy", embeddings)
        np.save(output_dir / "song_ids.npy", feature_ids)
        
        # Save CSV
        df = pd.DataFrame(songs)
        df.to_csv(output_dir / "song_features.csv", index=False)
        
        logger.info(f"✅ Exported {len(songs)} songs to {output_dir}")


# Singleton instance
_db_instance = None


def get_database() -> SongDatabase:
    """Get or create database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SongDatabase()
    return _db_instance


if __name__ == "__main__":
    # Test the database
    print("=" * 60)
    print("Song Database Test")
    print("=" * 60)
    
    db = get_database()
    print(f"\nDatabase path: {db.db_path}")
    print(f"Audio directory: {db.audio_dir}")
    print(f"Current song count: {db.get_song_count()}")
    
    # Test import from existing numpy files
    data_dir = Path(__file__).parent.parent / "data"
    if (data_dir / "song_features.npy").exists():
        print("\nImporting existing data...")
        db.import_from_numpy(
            str(data_dir / "song_features.npy"),
            str(data_dir / "song_embeddings.npy"),
            str(data_dir / "song_ids.npy"),
            str(data_dir / "song_features.csv")
        )
        print(f"Song count after import: {db.get_song_count()}")
