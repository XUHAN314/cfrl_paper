# Experiment Design

## 1. Main table

Table 1 compares the three stages against the ELF base and AR and diffusion.

- Maybe we can explicitly show the gain against the base SFT for each model. GPT on KK is already very good (92.9).
- **Conclusion:** JPC-2D improves its base model the best and the most consistently compared with Stages 1 and 2, as well as the AR and diffusion baselines.

## 2. Why pathwise estimator over score function (from Stage 1 to Stage 2)

We need to support the claim in the paper that the GRPO gradient has noise spread over millions of coordinates, making its variance large, whereas the pathwise estimator replaces this noise with the gradient through the chain itself.

Frame the evidence as two questions:

### Can the gradient direction be estimated reliably?

The right panel of Figure 1 measures reproducibility and variance. It supports the claim that GRPO has high variance whereas the pathwise estimator is reproducible.

- The GRPO curve stays close to 0, meaning noise dominates and the gradient signal is negligible relative to its variance. The 2D-critic curve stays close to 1, meaning both halves recover the same gradient direction.
- The GPT-GRPO line supports the claim that score-function RL works better when its score terms correspond to discrete token decisions.
- It does not establish that JPC points in the right direction: a reproducible gradient can still be biased. High half-split cosine means low variance and reproducibility, not low bias.

### Does the reproducible direction actually increase verifier reward?

- Table 1 (left) gives end-to-end evidence, but a targeted intervention would make the argument substantially stronger.
- Maybe add one small experiment for a matched one-step gradient intervention: start from the same base checkpoint; compute GRPO and JPC gradients from the same number of rollouts and the same sampled flow step; normalize both gradients to the same parameter-update norm; apply one small update in each direction; estimate the change in verifier reward using many evaluation rollouts, ideally with common prompts and common random noise; and repeat over several independently sampled training batches.

## 3. Why a 2D critic over a 1D critic (from Stage 2 to Stage 3)

We need to support the claim that assigning credit along both flow step and token position is better than assigning credit only along flow step.

- Figure 2 tells a good story, especially its right panel.
- We also need more in-depth experiments. The right panel of Figure 2 swaps the critic after 250 iterations, which is a good start. Sudoku may provide a useful example of how mean-then-head and head-then-mean affect the final answer. Starting from the policy trained for 250 steps with the 2D critic, continue training briefly with either the 2D or scalar critic and show how decoded tokens change across flow steps. This could demonstrate whether the 2D critic fixes particular tokens while the scalar critic changes the sequence more globally.

### Tier 0: Gradient-field autopsy

Do this first without training. Test the hypothesis that the 2D critic updates specific tokens, whereas the scalar critic updates the whole chain without targeting specific tokens.

Take matched states $y_k$ from identical rollouts of the shared policy, compute per-cell gradient norms $\lVert \partial V_\phi / \partial y_{k,i} \rVert$ under both critics, and compare them with Sudoku's per-cell oracle correctness.

Metrics:

- AUC of per-cell gradient mass versus cell wrongness.
- Fraction of total gradient norm assigned to wrong cells, compared with the fraction of cells that are wrong.

Prediction: the 2D critic concentrates gradient mass on wrong cells, whereas the scalar critic spreads it. The scalar critic's field is one shared feature direction projected through each cell's Jacobian.

### Tier 1: Training probe with three confounds controlled

The proposed structure is right: start from the shared 2D-at-250 checkpoint, branch into a few updates with the 2D critic versus a few updates with the scalar critic, and compare outputs. It needs the following controls:

1. **Common random numbers.** Compare the pre- and post-update policies on the same evaluation puzzles with the same noise seeds ($\varepsilon_0$ and all churn draws). A few iterations at learning rate $10^{-6}$ move $\theta$ very little; without matched noise, sampling stochasticity may overwhelm the parameter-induced difference.
2. **Matched parameter displacement, not matched iterations.** The scalar recipe uses $3\times10^{-6}$, whereas the 2D recipe uses $10^{-6}$. Running both for the same number of iterations would confound critic design with step size. Either use the same learning rate and compare at equal $\lVert\Delta\theta\rVert$, or report edit metrics as a function of $\lVert\Delta\theta\rVert$.
3. **Freeze both critics during the probe.** This avoids the training asymmetry in the code: the scalar trunk trains during joint updates, whereas only the 2D head trains. For a short probe, critic staleness should be negligible, and freezing isolates which gradient field steers the policy.

Using Sudoku's unique-solution ground truth, the ideal edit changes each wrong cell to its unique correct value and leaves correct cells untouched. For cells whose decoded token changes between the pre- and post-update policies, measure:

- **Edit precision:** fraction of edited cells that were initially wrong (selectivity, or whether the method "fixes tokens").
- **Edit recall:** fraction of wrong cells edited.
- **Preservation:** fraction of previously correct cells left unchanged.
- **Global-rewrite rate:** fraction of samples in which more than a chosen percentage of cells changes.

Prediction: the 2D critic has high precision and preservation, whereas the scalar critic has lower precision and more global rewrites. Because the scalar critic collapses on Sudoku, report the fix-versus-break split so that selective versus global editing is not conflated with improving versus degrading the output.

### Tier 2: Flow-step visualization

Decoding $\hat{x}_k$ at every step before and after each update is a byproduct of the Tier 1 rollouts. It provides a qualitative panel: for one held-out puzzle, show a $K\times81$ difference heatmap (unchanged, fixed to correct, or broken to wrong) for each critic. If the hypothesis holds, the 2D panel should show sparse corrections, whereas the scalar panel should show diffuse changes. This pairs naturally with the existing critic-map figure style.

### Tier 3: Recommended experiment

Start both branches from exactly the same policy checkpoint at iteration 250:

- Branch A continues with the 2D critic.
- Branch B switches to the scalar critic.
- Refit both critics identically on their respective fresh rollouts.
- Use the same prompts, Gaussian noise, flow steps, optimizer settings, and number of updates.

At several later training iterations, decode intermediate flow states for a fixed Sudoku evaluation set using the same sampling noise. Relative to the ground-truth grid, classify every position's change from the iteration-250 policy:

| Before update | After update | Interpretation |
|---|---|---|
| wrong | correct | useful repair |
| correct | wrong | collateral damage |
| wrong | wrong, different token | unsuccessful revision |
| correct | correct | preservation |

Then report:

$$
\text{repair rate}
=
\frac{\#(\text{wrong}\rightarrow\text{correct})}
{\#\text{ initially wrong}},
$$

$$
\text{damage rate}
=
\frac{\#(\text{correct}\rightarrow\text{wrong})}
{\#\text{ initially correct}},
$$

and perhaps

$$
\text{repair precision}
=
\frac{\#(\text{wrong}\rightarrow\text{correct})}
{\#\text{ positions whose decoded token changes}}.
$$

Plot these over both:

- **Actor-training iterations**, showing what continued optimization changes.
- **Flow steps**, showing when repairs emerge during denoising.

This directly tests whether JPC-2D produces more targeted semantic improvements rather than merely changing more tokens.

#### Suggested figure

A strong three-panel figure would be:

1. **Left:** held-out accuracy after the iteration-250 critic swap, which is already available.
2. **Middle:** repair and damage rates over subsequent actor updates for the scalar and 2D critics.
3. **Right:** one representative Sudoku trajectory with cells classified as repaired, damaged, unchanged correct, or still incorrect.

For the example, show the same prompt and noise at iteration 250 and after a fixed number of updates under each critic. Display only a few informative flow steps rather than every step.

#### A more direct architecture diagnostic

To isolate mean-then-head versus head-then-mean from actor training, evaluate localized perturbations on identical intermediate states:

1. Take a valid or partially correct Sudoku state.
2. Perturb one cell embedding.
3. Measure the change in each critic's value and actor gradient at every position.
4. Repeat over many states and cells.

Useful metrics include:

- Gradient mass on the perturbed position.
- Gradient mass on constraint-related cells in the same row, column, and box.
- Gradient mass on unrelated cells.
- Whether a gradient step restores the corrupted cell without damaging correct cells.

This is mechanistically cleaner, but the architecture may not exhibit strong localization because transformer attention couples positions. Treat it as an exploratory diagnostic rather than the primary experiment.

## 4. Design of joint training

Table 2 could move some variants to the critic ablation. Here, we only need to support the joint-training claim.

- My reading of the code: scalar JPC jointly trains the denoiser and head, but JPC-2D trains only the head during joint training. Do we have an experiment supporting this design choice? Table 3, where the head is refit, should be discussed here.

## 5. Why the loop?

The existing "the loop" section looks good.

## 6. Ablation

- Table 3: maybe move the learning-rate results to the appendix. The other ablations are useful: objective, pooling, and sampler. The head-refitting result could be discussed in Section 4 on joint training.
- Add the critic choices from Table 2, excluding random initialization.

For all plots, put the legend in the plot instead of explaining it in the caption.
