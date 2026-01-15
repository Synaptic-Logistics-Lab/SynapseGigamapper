"""
ESM-C (Cambrian) Protein Encoder
Uses the new ESM package with ESMC.from_pretrained() API

This is the encoder used for fine-tuning ESM-C 600M for synapse localization prediction.
Requires Python >= 3.10 and the new ESM package from EvolutionaryScale.
"""

import torch
import torch.nn as nn
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from protgps.models.abstract import AbstractModel
from protgps.utils.registry import register_object


@register_object("esmc_encoder", "model")
class ESMCEncoder(AbstractModel):
    """
    ESM-C (Cambrian) protein language model encoder

    Uses the new ESM package released December 2024
    Model: ESMC 600M - approaching ESM-2 3B performance

    Key specifications:
    - Embedding dim: 1152 (vs 320 for ESM2-8M)
    - Parameters: ~600M (vs 8M for ESM2-8M)
    - Layers: 36 transformer blocks
    - API: New ESMProtein interface
    - Auto-downloads weights from EvolutionaryScale
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ESM-C model
        print(f"Loading ESM-C model: {args.esm_name}")
        print(f"  Device: {self.device}")
        print(f"  Freeze encoder: {args.freeze_esm}")

        # Load model and move to device
        self.model = ESMC.from_pretrained(args.esm_name).to(self.device)

        # Freeze encoder if specified
        if args.freeze_esm:
            print("  Freezing ESM-C encoder (only MLP will be trained)")
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("  Fine-tuning ESM-C encoder (all parameters trainable)")
            self.model.train()

        # Get model configuration
        if hasattr(self.model, 'config'):
            self.hidden_size = getattr(self.model.config, 'hidden_size', 1152)
            self.num_layers = getattr(self.model.config, 'num_hidden_layers', None)
            print(f"  Hidden size: {self.hidden_size}")
            if self.num_layers:
                print(f"  Num layers: {self.num_layers}")
        else:
            # Default for esmc_600m
            self.hidden_size = 1152
            print(f"  Hidden size: {self.hidden_size} (default)")

        # Which layer to extract representations from
        self.repr_layer = args.esm_hidden_layer if hasattr(args, 'esm_hidden_layer') else -1
        print(f"  Representation layer: {self.repr_layer}")

        # Whether to output per-residue embeddings
        self.output_residue_hiddens = getattr(args, 'output_residue_hiddens', False)

        print("ESM-C encoder initialized successfully!")

    def forward(self, sequences, tokens=False, soft=False):
        """
        Forward pass through ESM-C encoder

        Args:
            sequences: List of protein sequence strings
                       e.g., ["MKTVRQERLK...", "MALWMRLLPL..."]
            tokens: bool (ignored, for compatibility with ESM2 interface)
            soft: bool (ignored, for compatibility with ESM2 interface)

        Returns:
            dict with keys:
                - "hidden": (batch_size, hidden_dim) - mean-pooled sequence embeddings
                - "residues": (batch_size, seq_len, hidden_dim) - per-residue embeddings (optional)
        """
        output = {}

        # Process each protein individually
        # ESM-C doesn't support batching at the ESMProteinTensor level,
        # so we process one at a time and batch the final embeddings
        mean_embeddings_list = []
        residue_embeddings_list = []

        for seq in sequences:
            # Convert sequence to ESMProtein object
            protein = ESMProtein(sequence=seq)

            # Encode protein to ESMProteinTensor
            protein_tensor = self.model.encode(protein)

            # Get embeddings for this protein
            # Note: model.logits() expects ESMProteinTensor, not raw tensors
            if self.args.freeze_esm:
                with torch.no_grad():
                    result = self.model.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_embeddings=True)
                    )
            else:
                result = self.model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )

            # Extract embeddings for this protein
            # Shape: (1, seq_len, hidden_dim)
            embeddings = result.embeddings

            # Mean pooling over sequence length to get single vector
            mean_emb = embeddings.mean(dim=1).squeeze(0)  # (hidden_dim,)

            # CRITICAL: Convert to float32 for MLP compatibility
            # ESM-C outputs BFloat16, but downstream layers expect Float32
            mean_emb = mean_emb.float()

            mean_embeddings_list.append(mean_emb)

            # Optionally store per-residue embeddings
            if self.output_residue_hiddens:
                residue_embeddings_list.append(embeddings)

        # Stack all mean embeddings into batch
        # Shape: (batch_size, hidden_dim)
        mean_embeddings = torch.stack(mean_embeddings_list)

        output["hidden"] = mean_embeddings

        # Optionally return per-residue embeddings
        # Note: These will be variable length, so we return as list
        if self.output_residue_hiddens:
            output["residues"] = residue_embeddings_list

        return output

    def get_embedding_dim(self):
        """Return the embedding dimension (1152 for esmc_600m)"""
        return self.hidden_size

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args for ESM-C encoder

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--pretrained_hub_dir",
            type=str,
            default="checkpoints/esmc",
            help="directory where ESM-C pretrained models are cached",
        )
        parser.add_argument(
            "--esm_name",
            type=str,
            default="esmc_600m",
            help="ESM-C model name (esmc_300m or esmc_600m)",
        )
        parser.add_argument(
            "--freeze_esm",
            action="store_true",
            help="freeze ESM-C encoder weights during training",
        )
        parser.add_argument(
            "--esm_hidden_layer",
            type=int,
            default=-1,
            help="which ESM-C layer to extract embeddings from (-1 = last layer)",
        )
        parser.add_argument(
            "--output_residue_hiddens",
            action="store_true",
            help="output per-residue embeddings in addition to mean embedding",
        )


# Register alternative names for backward compatibility
@register_object("esmc_600m", "model")
class ESMC600M(ESMCEncoder):
    """Alias for ESMCEncoder with esmc_600m model"""
    pass
