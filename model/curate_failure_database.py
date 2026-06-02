import os
import sys
import urllib.request
import urllib.parse
import gzip
import csv
import io
import json
import sqlite3
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Local fast-lookup for drug names to SMILES to minimize PubChem queries
KNOWN_DRUG_SMILES = {
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "haloperidol": "O=C(c1ccc(F)cc1)CCCN2CCC(n3c(=O)[nH]c4ccccc43)CC2",
    "nicotine": "CN1CCCC1c2cccnc2",
    "caffeine": "CN1C(=O)N(C)c2ncn(C)c2C1=O",
    "terfenadine": "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
    "astemizole": "COc1ccc(cc1)C2N(CCN2CCc3ccc(F)cc3)c4ccc(F)cc4",
    "troglitazone": "CC1Oc2ccc(CC3SC(=O)NC3=O)cc2Oc4c1c(C)c(O)c(C)c4C",
    "diclofenac": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",
    "sildenafil": "CCCC1=NN(C)C2=C1N=C(C3=C(OCCN)C=CC(=C3)S(=O)(=O)N4CCN(C)CC4)NC2=O",
    "valproic acid": "CCCC(CCC)C(=O)O",
    "amiodarone": "CCCCC1=C(C2=CC(I)=C(OCCN(CC)CC)C(I)=C2)OC3=CC=CC=C13",
    "cisapride": "COc1cc(Cl)c(N)cc1C(=O)NC2CCN(CCCOc3ccc(F)cc3)C(OC)C2",
    "sertindole": "Fc1ccc(cc1)C2CCN(CCN3C(=O)Nc4ccccc43)CC2",
    "thioridazine": "CN1CCCC1CCN2c3ccccc3Sc4ccc(SC)cc42",
    "digitoxin": "CC(=O)OC1C2CC3C(C(C2)O)C4(C3CC(O4)C5=CC(=O)OC5)C",
    "sarin": "CC(C)OP(=O)(F)OC(C)C",
    "cocaine": "CN1C2CCC1C(C(=O)OC)C(OC(=O)c3ccccc3)C2",
    "morphine": "CN1CCC23C4c5ccc(O)c2c5OC3C(O)C=CC14",
    "warfarin": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
    "lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    "digoxin": "CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CCC7(C)C(C8=CC(=O)OC8)CCC67O)C4)OC3)OC2)CC(O)C1O",
    "rosiglitazone": "CN(CCN1c2ccc(CC3SC(=O)NC3=O)cc2)c4ncccn4",
    "nitrobenzene": "O=N(=O)c1ccccc1",
    "1-chloro-4-nitrobenzene": "O=N(=O)c1ccc(Cl)cc1",
    "2-naphthylamine": "Nc1ccc2ccccc2c1",
    "musk xylene": "Cc1c(N(=O)=O)cc(C)c(N(=O)=O)c1C(C)(C)C",
    "benzo[a]pyrene": "c1ccc2c(c1)ccc3c4ccc5ccccc5c4c6ccc2c3c6",
    "2-acetylaminofluorene": "CC(=O)Nc1ccc2c(c1)Cc3ccccc32",
    "nitrofurantoin": "O=C1NC(=O)CN1/N=C/c2ccc(o2)[N+](=O)[O-]",
    "dianisidine": "COc1cc(ccc1N)c2ccc(c(c2)OC)N"
}

# Curation rules of known safety liabilities to augment prediction labels
KNOWN_HERG_BLOCKERS = {"haloperidol", "terfenadine", "astemizole", "cisapride", "sertindole", "thioridazine", "amiodarone", "imipramine", "quinidine", "dofetilide", "sotalol", "chlorpromazine"}
KNOWN_DILI_TOX = {"acetaminophen", "paracetamol", "troglitazone", "diclofenac", "valproic acid", "amiodarone", "rosiglitazone", "ketoconazole", "isoniazid", "methotrexate", "carbamazepine"}
KNOWN_AMES_MUTAGENS = {"nitrobenzene", "1-chloro-4-nitrobenzene", "2-naphthylamine", "musk xylene", "benzo[a]pyrene", "2-acetylaminofluorene", "nitrofurantoin", "dianisidine"}

def resolve_name_to_smiles(name):
    """
    Resolves drug or chemical name to a canonical SMILES string.
    Checks local fast-lookup dictionary and falls back dynamically to PubChem REST API.
    """
    clean_name = name.strip().lower()
    
    # 1. Local fast-lookup
    if clean_name in KNOWN_DRUG_SMILES:
        return KNOWN_DRUG_SMILES[clean_name]
        
    # 2. Remote NIH PubChem PUG REST API query
    try:
        encoded_name = urllib.parse.quote(clean_name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/JSON"
        res = requests.get(url, timeout=3.0)
        if res.status_code == 200:
            data = res.json()
            smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            return smiles
    except Exception as e:
        # Silently fail, name cannot be resolved
        pass
        
    return None

def fetch_clinical_trials_rejections():
    """
    Queries the live ClinicalTrials.gov API v2 for interventional studies
    that are terminated or withdrawn due to safety/toxicity, extracting
    their drug interventions and resolving them to SMILES.
    """
    print("\n--- [1/4] Ingesting Live clinical rejections from ClinicalTrials.gov REST API ---")
    
    # We query terminated or withdrawn trials with advanced Essie filters targeting WhyStopped reasons
    url = (
        "https://clinicaltrials.gov/api/v2/studies"
        "?filter.overallStatus=TERMINATED,WITHDRAWN"
        "&filter.advanced=AREA[WhyStopped]toxicity OR AREA[WhyStopped]safety OR AREA[WhyStopped]adverse OR AREA[WhyStopped]side effect"
        "&fields=NCTId,BriefTitle,WhyStopped,ArmsInterventionsModule"
        "&pageSize=80"
    )
    
    records = []
    try:
        res = requests.get(url, timeout=10.0)
        if res.status_code != 200:
            print(f"Failed to fetch clinical trials: HTTP {res.status_code}")
            return []
            
        data = res.json()
        studies = data.get("studies", [])
        print(f"Retrieved {len(studies)} safety-terminated clinical trials.")
        
        resolved_count = 0
        for study in studies[:15]:
            prot = study.get("protocolSection", {})
            nct_id = prot.get("identificationModule", {}).get("nctId", "N/A")
            title = prot.get("identificationModule", {}).get("briefTitle", "N/A")
            why_stopped = prot.get("statusModule", {}).get("whyStopped", "N/A")
            
            interventions = prot.get("armsInterventionsModule", {}).get("interventions", [])
            
            for intr in interventions:
                if intr.get("type") == "DRUG":
                    drug_name = intr.get("name", "").strip()
                    if drug_name and len(drug_name) > 2:
                        smiles = resolve_name_to_smiles(drug_name)
                        if smiles:
                            # Standardize using RDKit
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                                records.append({
                                    "smiles": canon_smiles,
                                    "name": drug_name,
                                    "failed_clinical_trial": 1,
                                    "herg_blocker": 1 if drug_name.lower() in KNOWN_HERG_BLOCKERS else 0,
                                    "dili_hepatotox": 1 if drug_name.lower() in KNOWN_DILI_TOX else 0,
                                    "ames_mutagen": 1 if drug_name.lower() in KNOWN_AMES_MUTAGENS else 0,
                                    "ld50_high_tox": 0,
                                    "clinical_reason": f"Clinical Trial {nct_id} terminated: '{why_stopped}'",
                                    "source": f"ClinicalTrials.gov ({nct_id})"
                                })
                                resolved_count += 1
                                break # Get one drug per study to prevent duplications
                                
        print(f"Successfully resolved {resolved_count} unique drug structures to SMILES.")
    except Exception as e:
        print(f"Error querying ClinicalTrials.gov API: {e}")
        
    return records

def download_clintox():
    """
    Downloads and parses MoleculeNet ClinTox dataset from DeepChem S3.
    """
    print("\n--- [2/4] Downloading ClinTox dataset from DeepChem S3 ---")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    
    records = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                csv_data = uncompressed.read().decode('utf-8')
                
        reader = csv.DictReader(io.StringIO(csv_data))
        for row in reader:
            smiles = row.get("smiles", "").strip()
            fda_approved = row.get("FDA_APPROVED", "").strip()
            ct_tox = row.get("CT_TOX", "").strip()
            
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                    failed_ct = 1 if ct_tox == "1" else 0
                    
                    # Estimate name if possible or use formula as placeholder
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    name = f"ClinTox compound ({formula})"
                    
                    # Match known labels
                    herg = 0
                    dili = 0
                    ames = 0
                    
                    # Attempt to resolve common structures dynamically
                    for k, v in KNOWN_DRUG_SMILES.items():
                        if Chem.MolFromSmiles(v) and Chem.MolToSmiles(Chem.MolFromSmiles(v), canonical=True) == canon_smiles:
                            name = k.capitalize()
                            herg = 1 if k in KNOWN_HERG_BLOCKERS else 0
                            dili = 1 if k in KNOWN_DILI_TOX else 0
                            ames = 1 if k in KNOWN_AMES_MUTAGENS else 0
                            break
                            
                    records.append({
                        "smiles": canon_smiles,
                        "name": name,
                        "failed_clinical_trial": failed_ct,
                        "herg_blocker": herg,
                        "dili_hepatotox": dili,
                        "ames_mutagen": ames,
                        "ld50_high_tox": 0,
                        "clinical_reason": "ClinTox clinical trial safety failure study" if failed_ct else "ClinTox FDA-approved chemical control",
                        "source": "MoleculeNet ClinTox"
                    })
        print(f"Successfully loaded {len(records)} ClinTox records.")
    except Exception as e:
        print(f"Failed to fetch ClinTox dataset: {e}")
        
    return records

def download_sider():
    """
    Downloads and parses SIDER dataset from DeepChem S3.
    """
    print("\n--- [3/4] Downloading SIDER adverse drug reactions dataset from DeepChem S3 ---")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
    
    records = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                csv_data = uncompressed.read().decode('utf-8')
                
        reader = csv.DictReader(io.StringIO(csv_data))
        for row in reader:
            smiles = row.get("smiles", "").strip()
            hepato = row.get("Hepatobiliary disorders", "").strip()
            cardio = row.get("Cardiac disorders", "").strip()
            
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                    dili_val = 1 if hepato == "1" else 0
                    herg_val = 1 if cardio == "1" else 0
                    
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    name = f"SIDER drug ({formula})"
                    
                    for k, v in KNOWN_DRUG_SMILES.items():
                        if Chem.MolFromSmiles(v) and Chem.MolToSmiles(Chem.MolFromSmiles(v), canonical=True) == canon_smiles:
                            name = k.capitalize()
                            dili_val = 1
                            break
                            
                    records.append({
                        "smiles": canon_smiles,
                        "name": name,
                        "failed_clinical_trial": 0,
                        "herg_blocker": herg_val,
                        "dili_hepatotox": dili_val,
                        "ames_mutagen": 0,
                        "ld50_high_tox": 0,
                        "clinical_reason": f"SIDER adverse event: hepato_risk={dili_val}, cardiac_risk={herg_val}",
                        "source": "DeepChem SIDER"
                    })
        print(f"Successfully loaded {len(records)} SIDER records.")
    except Exception as e:
        print(f"Failed to fetch SIDER dataset: {e}")
        
    return records

def download_tox21():
    """
    Downloads and parses Tox21 dataset from DeepChem S3.
    """
    print("\n--- [4/4] Downloading Tox21 in vitro pathway assays from DeepChem S3 ---")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    
    records = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                csv_data = uncompressed.read().decode('utf-8')
                
        reader = csv.DictReader(io.StringIO(csv_data))
        for row in reader:
            smiles = row.get("smiles", "").strip()
            p53 = row.get("SR-p53", "").strip()
            mitochondria = row.get("SR-MMP", "").strip()
            
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                    
                    # Highlight high acute hazard if cell-death (p53) or mitochondrial stress (MMP) is active
                    high_tox = 1 if (p53 in ["1", "1.0"] or mitochondria in ["1", "1.0"]) else 0
                    
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    name = f"Tox21 compound ({formula})"
                    
                    records.append({
                        "smiles": canon_smiles,
                        "name": name,
                        "failed_clinical_trial": 0,
                        "herg_blocker": 0,
                        "dili_hepatotox": 0,
                        "ames_mutagen": 0,
                        "ld50_high_tox": high_tox,
                        "clinical_reason": f"Tox21 cellular assay: p53_active={p53}, mmp_disrupted={mitochondria}",
                        "source": "DeepChem Tox21"
                    })
        print(f"Successfully loaded {len(records)} Tox21 records.")
    except Exception as e:
        print(f"Failed to fetch Tox21 dataset: {e}")
        
    return records

def build_database():
    """
    Main pipeline coordinator. Merges all downloaded datasets, de-duplicates
    canonical SMILES, cleans properties, and outputs the SQLite and CSV database.
    """
    # 1. Fetch from all sources
    all_raw = []
    all_raw.extend(fetch_clinical_trials_rejections())
    all_raw.extend(download_clintox())
    all_raw.extend(download_sider())
    all_raw.extend(download_tox21())
    
    print(f"\n--- Aggregating and de-duplicating {len(all_raw)} total compounds ---")
    
    # 2. De-duplicate by canonical SMILES, merging toxicity profiles
    merged_db = {}
    
    for rec in all_raw:
        smiles = rec["smiles"]
        if smiles not in merged_db:
            merged_db[smiles] = rec
        else:
            # Merge endpoint flags (take max logical or)
            existing = merged_db[smiles]
            existing["failed_clinical_trial"] = max(existing["failed_clinical_trial"], rec["failed_clinical_trial"])
            existing["herg_blocker"] = max(existing["herg_blocker"], rec["herg_blocker"])
            existing["dili_hepatotox"] = max(existing["dili_hepatotox"], rec["dili_hepatotox"])
            existing["ames_mutagen"] = max(existing["ames_mutagen"], rec["ames_mutagen"])
            existing["ld50_high_tox"] = max(existing["ld50_high_tox"], rec["ld50_high_tox"])
            
            # Combine reasons and names
            if rec["failed_clinical_trial"] == 1 and existing["failed_clinical_trial"] == 0:
                existing["clinical_reason"] = rec["clinical_reason"]
            if rec["name"] and not rec["name"].startswith("ClinTox") and not rec["name"].startswith("SIDER") and not rec["name"].startswith("Tox21"):
                existing["name"] = rec["name"]
                
            existing["source"] = existing["source"] + " | " + rec["source"]
            
    # 3. Inject explicit known toxicological libraries to enrich structural endpoints
    for name, smiles in KNOWN_DRUG_SMILES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canon_smiles = Chem.MolToSmiles(mol, canonical=True)
            if canon_smiles not in merged_db:
                merged_db[canon_smiles] = {
                    "smiles": canon_smiles,
                    "name": name.capitalize(),
                    "failed_clinical_trial": 1 if name in KNOWN_HERG_BLOCKERS or name in KNOWN_DILI_TOX else 0,
                    "herg_blocker": 1 if name in KNOWN_HERG_BLOCKERS else 0,
                    "dili_hepatotox": 1 if name in KNOWN_DILI_TOX else 0,
                    "ames_mutagen": 1 if name in KNOWN_AMES_MUTAGENS else 0,
                    "ld50_high_tox": 0,
                    "clinical_reason": f"Known reference compound: cardiotox={name in KNOWN_HERG_BLOCKERS}, hepatotox={name in KNOWN_DILI_TOX}, mutagen={name in KNOWN_AMES_MUTAGENS}",
                    "source": "Curated Toxicology Reference"
                }
            else:
                existing = merged_db[canon_smiles]
                existing["name"] = name.capitalize()
                if name in KNOWN_HERG_BLOCKERS: existing["herg_blocker"] = 1
                if name in KNOWN_DILI_TOX: existing["dili_hepatotox"] = 1
                if name in KNOWN_AMES_MUTAGENS: existing["ames_mutagen"] = 1
                
    # 4. Save results to relational SQLite and CSV database
    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", "failure_database.sqlite")
    csv_path = os.path.join("data", "failure_database.csv")
    
    # SQLite Output
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop and recreate failures table
    cursor.execute("DROP TABLE IF EXISTS failures")
    cursor.execute("""
        CREATE TABLE failures (
            smiles TEXT PRIMARY KEY,
            name TEXT,
            failed_clinical_trial INTEGER,
            herg_blocker INTEGER,
            dili_hepatotox INTEGER,
            ames_mutagen INTEGER,
            ld50_high_tox INTEGER,
            clinical_reason TEXT,
            source TEXT
        )
    """)
    
    # Insert de-duplicated records
    rows_to_insert = [
        (
            rec["smiles"],
            rec["name"],
            rec["failed_clinical_trial"],
            rec["herg_blocker"],
            rec["dili_hepatotox"],
            rec["ames_mutagen"],
            rec["ld50_high_tox"],
            rec["clinical_reason"],
            rec["source"]
        )
        for rec in merged_db.values()
    ]
    
    cursor.executemany("""
        INSERT INTO failures (
            smiles, name, failed_clinical_trial, herg_blocker, dili_hepatotox, ames_mutagen, ld50_high_tox, clinical_reason, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    
    conn.commit()
    conn.close()
    
    # CSV Output
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "smiles", "name", "failed_clinical_trial", "herg_blocker", "dili_hepatotox", "ames_mutagen", "ld50_high_tox", "clinical_reason", "source"
        ])
        writer.writeheader()
        for rec in merged_db.values():
            writer.writerow(rec)
            
    print(f"\n=======================================================")
    print(f"DATABASE CURATION PIPELINE SUCCESSFUL!")
    print(f"Total Unique De-duplicated Compounds Ingested: {len(merged_db)}")
    print(f"  - Clinical rejections (ClinicalTrials.gov / ClinTox): {sum(1 for r in merged_db.values() if r['failed_clinical_trial'] == 1)}")
    print(f"  - Cardiotoxic safety concerns (hERG active): {sum(1 for r in merged_db.values() if r['herg_blocker'] == 1)}")
    print(f"  - Hepatotoxic concerns (DILI active): {sum(1 for r in merged_db.values() if r['dili_hepatotox'] == 1)}")
    print(f"  - Mutagenic concerns (Ames active): {sum(1 for r in merged_db.values() if r['ames_mutagen'] == 1)}")
    print(f"  - High acute cellular toxicity (LD50 active): {sum(1 for r in merged_db.values() if r['ld50_high_tox'] == 1)}")
    print(f"Output files stored successfully under:")
    print(f"  - Relational Database: {db_path}")
    print(f"  - Clean Dataset Sheet: {csv_path}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    build_database()
