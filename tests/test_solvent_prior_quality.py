#!/usr/bin/env python
"""
Solvent Prior Quality Validation

Loads a SATURN prior checkpoint, samples SMILES, and reports quality metrics.
Used to compare the fine-tuned solvent prior against the zinc-250k baseline.

Usage:
    python tests/test_solvent_prior_quality.py                          # auto-detect best prior
    python tests/test_solvent_prior_quality.py --prior path/to/model.prior
    python tests/test_solvent_prior_quality.py --prior zinc              # explicit zinc baseline
    python tests/test_solvent_prior_quality.py --compare                 # side-by-side comparison
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SATURN_DIR = PROJECT_ROOT / "modules" / "drl4procsyn" / "molecular_design" / "saturn"
ZINC_PRIOR = SATURN_DIR / "experimental_reproduction" / "checkpoint_models" / "zinc-250k-mamba-epoch-50.prior"
SOLVENT_PRIOR = SATURN_DIR / "data" / "solvent-mamba.prior"
CHECKPOINT_DIR = SATURN_DIR / "data" / "solvent_prior_checkpoints"


def find_best_checkpoint():
    """Find the solvent-mamba.prior or latest checkpoint."""
    if SOLVENT_PRIOR.exists():
        return SOLVENT_PRIOR
    if CHECKPOINT_DIR.exists():
        priors = sorted(CHECKPOINT_DIR.glob("*.prior"))
        if priors:
            return priors[-1]
    return None


def load_and_sample(prior_path: str, n_samples: int = 10000, batch_size: int = 512,
                    device: str = "cpu"):
    """Load a prior and sample SMILES."""
    import torch
    from modules.drl4procsyn.molecular_design.saturn.models.generator import Generator

    print(f"Loading prior: {prior_path}")
    model = Generator.load_from_file(str(prior_path), device)
    model.set_mode("eval")

    print(f"Sampling {n_samples} SMILES (batch_size={batch_size})...")
    with torch.no_grad():
        smiles_list, _ = model.sample_smiles(num=n_samples, batch_size=batch_size)

    return smiles_list


def compute_metrics(smiles_list):
    """Compute quality metrics for a list of SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from modules.drl4procsyn.molecular_design.solvent_filter import passes_solvent_filter

    total = len(smiles_list)

    # Validity
    valid_mols = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))  # canonical

    n_valid = len(valid_mols)
    validity = n_valid / total if total > 0 else 0

    # Uniqueness (among valid)
    unique_smiles = set(valid_smiles)
    n_unique = len(unique_smiles)
    uniqueness = n_unique / n_valid if n_valid > 0 else 0

    # MW distribution
    mws = [Descriptors.ExactMolWt(mol) for mol in valid_mols]
    mw_arr = np.array(mws) if mws else np.array([0.0])

    # Heavy atom distribution
    n_heavy = [mol.GetNumHeavyAtoms() for mol in valid_mols]
    heavy_arr = np.array(n_heavy) if n_heavy else np.array([0])

    # Solvent filter pass rate
    n_pass_filter = sum(1 for smi in valid_smiles if passes_solvent_filter(smi))
    filter_rate = n_pass_filter / n_valid if n_valid > 0 else 0

    # Element distribution (top elements)
    element_counts = Counter()
    for mol in valid_mols:
        for atom in mol.GetAtoms():
            element_counts[atom.GetSymbol()] += 1

    return {
        "total": total,
        "valid": n_valid,
        "validity": validity,
        "unique": n_unique,
        "uniqueness": uniqueness,
        "mw_mean": float(np.mean(mw_arr)),
        "mw_std": float(np.std(mw_arr)),
        "mw_median": float(np.median(mw_arr)),
        "mw_p10": float(np.percentile(mw_arr, 10)),
        "mw_p90": float(np.percentile(mw_arr, 90)),
        "heavy_mean": float(np.mean(heavy_arr)),
        "heavy_std": float(np.std(heavy_arr)),
        "heavy_median": float(np.median(heavy_arr)),
        "filter_pass": n_pass_filter,
        "filter_rate": filter_rate,
        "top_elements": element_counts.most_common(10),
    }


def print_metrics(name, metrics):
    """Pretty-print metrics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Sampled:      {metrics['total']}")
    print(f"  Valid:         {metrics['valid']}  ({metrics['validity']:.1%})")
    print(f"  Unique:        {metrics['unique']}  ({metrics['uniqueness']:.1%})")
    print(f"  MW:            {metrics['mw_mean']:.1f} +/- {metrics['mw_std']:.1f}  "
          f"(median={metrics['mw_median']:.1f}, p10={metrics['mw_p10']:.1f}, p90={metrics['mw_p90']:.1f})")
    print(f"  Heavy atoms:   {metrics['heavy_mean']:.1f} +/- {metrics['heavy_std']:.1f}  "
          f"(median={metrics['heavy_median']:.0f})")
    print(f"  Solvent filter: {metrics['filter_pass']}/{metrics['valid']}  ({metrics['filter_rate']:.1%})")
    print(f"  Top elements:  {metrics['top_elements']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate SATURN solvent prior quality")
    parser.add_argument("--prior", type=str, default=None,
                        help="Path to .prior file, or 'zinc' for baseline")
    parser.add_argument("--compare", action="store_true",
                        help="Compare solvent prior vs zinc baseline side-by-side")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.compare:
        # Side-by-side comparison
        solvent_path = find_best_checkpoint()
        if solvent_path is None:
            print("ERROR: No solvent prior found. Run fine-tuning first.")
            sys.exit(1)
        if not ZINC_PRIOR.exists():
            print(f"ERROR: Zinc prior not found at {ZINC_PRIOR}")
            sys.exit(1)

        zinc_smiles = load_and_sample(ZINC_PRIOR, args.n_samples, args.batch_size, args.device)
        zinc_metrics = compute_metrics(zinc_smiles)
        print_metrics("Zinc-250k Baseline", zinc_metrics)

        solvent_smiles = load_and_sample(solvent_path, args.n_samples, args.batch_size, args.device)
        solvent_metrics = compute_metrics(solvent_smiles)
        print_metrics(f"Solvent Prior ({solvent_path.name})", solvent_metrics)

        # Delta summary
        print("--- Delta (Solvent - Zinc) ---")
        print(f"  Validity:      {solvent_metrics['validity'] - zinc_metrics['validity']:+.1%}")
        print(f"  Uniqueness:    {solvent_metrics['uniqueness'] - zinc_metrics['uniqueness']:+.1%}")
        print(f"  MW mean:       {solvent_metrics['mw_mean'] - zinc_metrics['mw_mean']:+.1f}")
        print(f"  Filter rate:   {solvent_metrics['filter_rate'] - zinc_metrics['filter_rate']:+.1%}")
    else:
        # Single prior evaluation
        if args.prior is None:
            prior_path = find_best_checkpoint()
            if prior_path is None:
                print("No solvent prior found, falling back to zinc baseline.")
                prior_path = ZINC_PRIOR
        elif args.prior.lower() == "zinc":
            prior_path = ZINC_PRIOR
        else:
            prior_path = Path(args.prior)

        if not prior_path.exists():
            print(f"ERROR: Prior not found at {prior_path}")
            sys.exit(1)

        smiles_list = load_and_sample(prior_path, args.n_samples, args.batch_size, args.device)
        metrics = compute_metrics(smiles_list)
        print_metrics(prior_path.name, metrics)


if __name__ == "__main__":
    main()
