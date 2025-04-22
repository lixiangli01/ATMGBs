# ATMGBs

**ATMGBs** (Attention Maps and Graph convolutional networks for Binding site prediction) is a novel deep learning framework for sequence-based prediction of protein-nucleic acid binding sites. It integrates multiple feature types and leverages attention maps from protein language models to simulate spatial relationships between residues — without requiring 3D structural information.

---

## 🔍 Overview

Protein-nucleic acid binding sites are vital for biological processes such as gene expression, replication, and transcription. While structure-based models have demonstrated strong performance, they rely heavily on the availability of high-quality 3D protein structures, which limits their scalability.

**ATMGBs** addresses this challenge by building a fully sequence-based model that approaches the performance of structure-based frameworks.

---

## 💡 Key Features

- **Multi-source embeddings**: Combines embeddings from **Prot-T5** and **ESM** protein language models to capture rich contextual features from sequences.
- **Attention map modeling**: Uses the attention maps from **Prot-T5** to simulate residue-residue relationships, mimicking structural proximity.
- **Physicochemical descriptors**: Incorporates amino acid physicochemical properties to enhance stability and interpretability.


---

## 🧠 Model Architecture

1. **Input**: Protein sequences
2. **Feature Extraction**:
   - Prot-T5 embeddings
   - ESM embeddings
   - Attention maps from Prot-T5
   - Physicochemical features
3. **Graph Construction**: Based on attention-derived residue relationships
4. **Graph Convolutional Network**: Deep feature learning over the residue graph
5. **Output**: Binding site prediction scores for each residue

