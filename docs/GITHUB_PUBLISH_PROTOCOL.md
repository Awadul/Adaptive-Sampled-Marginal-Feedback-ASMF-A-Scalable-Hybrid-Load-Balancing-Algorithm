# GitHub Publish Protocol (Project Submission)

This protocol follows a clean, academic repository style suitable for project submission and reproducibility checks.

## 1. Repository Naming

Recommended: `asmf-load-balancer-simulator`

## 2. Branching Strategy

- `main`: stable, reviewed branch.
- `dev`: integration branch for active work.
- feature branches: `feat/<topic>`.

## 3. Commit Convention

Use Conventional Commits:

- `feat: add ASMF routing engine`
- `test: add routing and feedback tests`
- `docs: add proof simulation protocol`
- `chore: update dependencies`

## 4. Pull Request Checklist

- [ ] Description of algorithmic changes
- [ ] Updated tests and pass status
- [ ] Simulation rerun if logic changed
- [ ] Documentation updates
- [ ] Reviewer assigned

## 5. Release Tagging

- Tag submission versions as `vX.Y.Z`.
- Example: `v1.0.0-submission`.

## 6. Required Artifacts for Submission

- Source code and tests
- `README.md` with setup and run steps
- `docs/PROOF_SIMULATIONS.md`
- Latest benchmark summary CSV and plots

## 7. Recommended Git Commands

```bash
git init
git add .
git commit -m "feat: initial ASMF simulator and benchmark suite"
git branch -M main
git remote add origin https://github.com/<user>/asmf-load-balancer-simulator.git
git push -u origin main
```

## 8. Optional CI (Suggested)

Set up GitHub Actions for:

- lint/test on push
- periodic benchmark sanity check

## 9. Citation Block (for README)

Add project citation metadata if required by your course or lab.
