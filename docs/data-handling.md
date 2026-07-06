# Data Handling

## Current State

- The existing `Data/Anatomy and Physiology 2e, J. Gordon Betts, Peter Desaix, Eddie Johnson.pdf` file is required by the current indexing flow.
- `scripts/index_data.py` still reads PDFs from `Data/`, so removing the file without adding a replacement source would break indexing.

## Recommendation

- Keep the current PDF locally for development until an external storage workflow is introduced.
- For larger future datasets, use an external bucket or mounted volume and load them at runtime.
- If binary datasets must remain in git temporarily, prefer Git LFS.

## Future Migration Path

- Add a dataset download or sync script.
- Store large PDFs outside the repository.
- Update the indexing script to read from a configurable dataset path.

