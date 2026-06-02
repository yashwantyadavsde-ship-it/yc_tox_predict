import os
import sqlite3
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class MolecularVectorDatabase:
    """
    A lightweight, zero-dependency, high-performance molecular vector database.
    Converts SMILES strings into 2,048-bit RDKit Morgan circular fingerprints
    and uses optimized NumPy calculations to run fast Tanimoto similarity searches.
    """
    def __init__(self, db_path="data/failure_database.sqlite", index_path="data/vector_index.npz"):
        self.db_path = db_path
        self.index_path = index_path
        
        # In-memory arrays to back similarity search
        self.fingerprints = None # numpy array of size [N, 2048]
        self.smiles = None       # list of size [N]
        self.names = None        # list of size [N]
        self.labels = None       # numpy array of size [N, 5] (clinical, herg, dili, ames, ld50)
        self.reasons = None      # list of size [N]
        self.sources = None      # list of size [N]
        
    def build_index(self):
        """
        Builds the molecular vector index by parsing SQLite records,
        generating Morgan fingerprints, and serializing representations to NPZ.
        Loads from serialized index automatically if it already exists.
        """
        if os.path.exists(self.index_path):
            print(f"Loading pre-computed molecular vector index from {self.index_path}...")
            try:
                data = np.load(self.index_path, allow_pickle=True)
                self.fingerprints = data["fingerprints"]
                self.smiles = data["smiles"].tolist()
                self.names = data["names"].tolist()
                self.labels = data["labels"]
                self.reasons = data["reasons"].tolist()
                self.sources = data["sources"].tolist()
                print(f"Successfully loaded vector index: {self.fingerprints.shape[0]} compounds.")
                return
            except Exception as e:
                print(f"Failed to load serialized index: {e}. Re-indexing database...")
                
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database {self.db_path} does not exist. Please run curate script first.")
            
        print("Indexing Clinical Failure Database into vector representations...")
        
        # Connect and retrieve all records
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT smiles, name, failed_clinical_trial, herg_blocker, dili_hepatotox, ames_mutagen, ld50_high_tox, clinical_reason, source FROM failures")
        rows = cursor.fetchall()
        conn.close()
        
        print(f"Total rows retrieved from SQLite: {len(rows)}")
        
        fp_list = []
        smiles_list = []
        names_list = []
        labels_list = []
        reasons_list = []
        sources_list = []
        
        for idx, row in enumerate(rows):
            smiles = row[0]
            name = row[1]
            labels = [row[2], row[3], row[4], row[5], row[6]]
            reason = row[7]
            source = row[8]
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    # Generate 2048-bit Morgan circular fingerprint (radius = 2)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    
                    # Convert to numpy binary array
                    arr = np.zeros((1,), dtype=np.int8)
                    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                    
                    fp_list.append(arr)
                    smiles_list.append(smiles)
                    names_list.append(name)
                    labels_list.append(labels)
                    reasons_list.append(reason)
                    sources_list.append(source)
                except Exception as e:
                    # Skip malformed compounds
                    pass
                    
        # Compile into arrays
        self.fingerprints = np.array(fp_list, dtype=np.bool_)
        self.smiles = smiles_list
        self.names = names_list
        self.labels = np.array(labels_list, dtype=np.int8)
        self.reasons = reasons_list
        self.sources = sources_list
        
        print(f"Molecular vector indexing complete: {self.fingerprints.shape[0]} / {len(rows)} successfully compiled.")
        
        # Serialize representations
        try:
            np.savez_compressed(
                self.index_path,
                fingerprints=self.fingerprints,
                smiles=np.array(self.smiles),
                names=np.array(self.names),
                labels=self.labels,
                reasons=np.array(self.reasons),
                sources=np.array(self.sources)
            )
            print(f"Vector index serialized to {self.index_path}")
        except Exception as e:
            print(f"Failed to serialize index file: {e}")
            
    def search(self, query_smiles, k=3):
        """
        Executes a real-time Tanimoto molecular similarity search against the vector index.
        Returns the top k nearest neighbors.
        """
        if self.fingerprints is None or len(self.fingerprints) == 0:
            self.build_index()
            
        mol = Chem.MolFromSmiles(query_smiles)
        if not mol:
            return []
            
        try:
            # Generate Morgan fingerprint for query
            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            query_vector = np.zeros((1,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(query_fp, query_vector)
            query_vector = np.array(query_vector, dtype=np.bool_)
            
            # Compute Tanimoto similarity using optimized vector bit operations:
            # T = intersection / union
            intersection = np.logical_and(self.fingerprints, query_vector).sum(axis=-1)
            union = np.logical_or(self.fingerprints, query_vector).sum(axis=-1)
            
            # Prevent divide by zero for compounds with no active bits
            union = np.where(union == 0, 1, union)
            similarities = intersection / union
            
            # Get sorted indices in descending order
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Package top k results
            results = []
            for idx in sorted_indices[:k]:
                sim = float(similarities[idx])
                results.append({
                    "smiles": self.smiles[idx],
                    "name": self.names[idx],
                    "tanimoto_similarity": round(sim, 4),
                    "failed_clinical_trial": int(self.labels[idx, 0]),
                    "herg_blocker": int(self.labels[idx, 1]),
                    "dili_hepatotox": int(self.labels[idx, 2]),
                    "ames_mutagen": int(self.labels[idx, 3]),
                    "ld50_high_tox": int(self.labels[idx, 4]),
                    "clinical_reason": self.reasons[idx],
                    "source": self.sources[idx]
                })
            return results
        except Exception as e:
            print(f"Error executing vector similarity search: {e}")
            return []
            
if __name__ == "__main__":
    # Test index building and search
    db = MolecularVectorDatabase()
    db.build_index()
    
    # Run test search for Aspirin
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    print(f"\nRunning similarity search for Aspirin ('{test_smiles}'):")
    res = db.search(test_smiles, k=3)
    for i, r in enumerate(res):
        print(f"[{i}] {r['name']} | Similarity: {r['tanimoto_similarity']*100:.2f}% | Source: {r['source']}")
        print(f"    Reason: {r['clinical_reason']}")
