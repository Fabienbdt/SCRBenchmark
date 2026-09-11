# Third-party provenance and licensing review

The repository contains adapted algorithms and preserved author implementations.
The existing project copyright notice is unchanged. A project-wide license must
be selected by the rights holders before presenting the software as open source.
This inventory records local evidence; a missing license file here does not
establish the license of the upstream project.

| Component | Local evidence | Follow-up before redistribution in a release |
| --- | --- | --- |
| `external/original_code/PARC` | A `LICENSE` file is included. | Preserve the license and record the exact upstream revision. |
| `external/original_code/scMAE-main` | README points to `CSUBioGroup/scMAE`; no standalone license is present in this copy. | Verify the license at the revision from which the code was taken. |
| `external/original_code/desc` | Author source and example data are present; no standalone license is present in this copy. | Establish the source revision and applicable code/data licenses. |
| `vendor/torchlars` | README declares v0.1.2, upstream `kakaobrain/torchlars`, Apache-2.0; setup metadata declares Apache. | Restore the exact upstream license and notices before distributing this component. |
| Adapted baselines in `src/scrbenchmark/algorithms` | Source references are documented in the implementation and method registry. | Include them in the provenance review; moving code does not remove attribution obligations. |

Large experiment bundles retain their existing locations. Removing them before
an alternative archive has been published and verified would break checkpoint
replay. A storage migration must record an immutable URL, size and SHA-256 for
each artifact and update the replay/download tests before removing Git copies.
History rewriting is a separate, coordinated operation after archive validation.
