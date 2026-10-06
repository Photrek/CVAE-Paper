# tex files/ — mistakes

- **Coupling defined as the asymptotic tail shape.** Papers from 2015–25 used κ = tail shape. That mixes the nonlinearity with the stretching parameter. The correct κ is the source of nonlinearity, shape / α (see the footnote in the Background section). Don't bring back the old definition or results that depend on it.
- **κ < 0 case of Eq. `partitionFunctionCoupledGaussian` (fixed 2026-10-06).** It had Γ(−1/κ − d/2)/Γ(−1/κ). Integrating the compact-support density gives Γ(1 − 1/κ − d/2)/Γ(1 − 1/κ). It was checked against a brute-force integral; at d = 2, Z = 2π|Σ|^{1/2} for every κ.
- **Tsallis q-statistics as the base framework.** It was rejected because q is a function of more basic parameters, and its "inverse scale" β is not the inverse of the scale (`nelsonOpenProblemsNonextensive2024`). Present Tsallis only as related work.
