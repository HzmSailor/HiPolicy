# Release changes

## Packaging and paths

- Created a self-contained release directory; the supplied V1/V2 trees are the
  untouched comparison source. Existing `AGENTS.md` is preserved.
- Kept both DP and DP3. Shared each policy package across benchmarks, retaining
  explicit compatibility branches for differing DP computations and buffers.
  Kept each benchmark's original dataset and trainer implementations.
- Added a working-directory-independent CLI for all four routes, explicit
  checkpoint selection, separate data/output roots, offline logging, configuration
  recording, asset download and original dependency-patch helpers.
- Replaced machine-specific data/video/checkpoint paths and a private service
  endpoint. Removed unused backup model files and unused third-policy model code.
  Kept simulation task bodies and language-generation utilities used at runtime.
- Collection writes seed/scene metadata into the selected data root. DP3 enables
  point-cloud collection/evaluation. Conversion stages output until successful
  completion, checks episode boundaries/array shapes and rejects existing datasets;
  V2 DP3 now supplies a rank-correct Zarr chunk shape. Converted numeric arrays,
  channel order and per-benchmark observation/action alignment are unchanged.
- Retained licenses and reference-only 16D configurations. Ignored generated
  assets, credentials, environments, logs, checkpoints and the paper PDF.
- Resumption preserves an explicitly selected output directory instead of
  restoring the previous machine's directory. Explicit resume configuration is
  checked before loading training state; optimizer and counter loading is unchanged.
- Removed Python 3.9-only AST inspection calls from the computation checker and
  normalized Python 3.8 subscript nodes for the stored fingerprints.

## Explicitly approved experiment changes

- DP: 600 epochs, batch 128, AdamW `lr=1e-4`, betas `[0.9,0.999]`, DDPM/100.
- DP3: 3,000 epochs, batch 128, the same AdamW settings, DDPM/100 instead of
  DDIM/10; retained `sample` prediction and 64-dimensional timestep embedding.
- Unified optional entropy guidance: off by default (one sample/short chunk);
  on uses **100 parallel candidates**, global entropy thresholds `-6` and `-5.5`,
  and executes candidate zero. The candidate batch expansion follows original
  V2 behavior. A corrected inverse reshape preserves observation identity for
  input batch sizes above one; simulation evaluates one observation at a time.
- Exposed `action_chunks` in prediction results, without introducing parameters.

## Preserved behavior

No changes to encoder/U-Net/attention definitions, conditioning, temporal sampling,
normalization, prediction targets, loss computation, EMA choices, optimizer update
ordering, gradient accumulation, or simulator dynamics beyond the approved
configuration/inference changes above. DP's benchmark-specific executable bodies
and DP3's shared bodies are checked against fingerprints extracted from the
original source. In particular, attention dimension conventions are preserved.
Full-size CPU comparisons also pass for all four routes: fixed-input actions,
losses, all gradient hashes and one AdamW update match exactly under the original
scheduler/optimizer settings. These checks use synthetic data and identity
normalizers, separate from the approved release defaults and EG interface tests.

## Outstanding external validation

Original complete environment exports, V1's omitted point-cloud extension source,
and original V2 extension revisions remain unavailable. Real experiment checkpoints,
GPU parity checks, simulator smoke runs and paper-scale training/evaluation are
required before claiming verified reproduction. See `VALIDATION.md`.
