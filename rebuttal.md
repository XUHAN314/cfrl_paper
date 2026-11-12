## Reviewer oAZU

We thank the reviewer for their positive assessment of our work and for recognizing the strengths of CFRL, particularly its application of pathwise policy optimization to continuous language generation and its ability to accommodate non-differentiable decoding and verification. We also appreciate the reviewer's acknowledgment that our gradient agreement and single-update experiments provide evidence of the critic's ability to guide effective policy optimization. We address the two concerns below.

> The within-ELF comparison lacks compute matching and clear attribution. Appendix D reports 4–6M rollout denoiser calls for CFRL-SVG versus 1.2M for CFRL-GRR (about 2M for SVG on Sudoku), with additional critic pre-training and online computation. The same-rollout gradient experiments support update quality but do not establish performance under equal total compute. A budget-matched comparison including critic costs is needed.

**Compute accounting.** Appendix D previously reported only rollout denoiser calls, which did not fully capture the computational costs. We have now added a comprehensive FLOPs accounting and compute-matched comparisons that include critic pre-training and online computation. We measure total training FLOPs across all network evaluations, including rollouts, actor updates, critic pre-training, and online critic updates. Following the standard parameter-based FLOPs approximation ([1]), we count a forward pass as one unit and a full backward pass as two units. We additionally approximate an input-gradient-only backward pass as one unit, since the critic backbone is frozen during RL. The full-run Table1 compute of CFRL-SVG is 1.2–2.7× that of the original CFRL-GRR, rather than the 3–5× suggested by rollout counts alone. The critic stage accounts for 14–42% of the total SVG compute.

**Compute-matched comparison.** We additionally trained CFRL-GRR using the same deployment sampler as CFRL-SVG (64 steps, or 32 for Sudoku), rather than its original 16-step sampler. This approximately doubles GRR's computational cost while keeping all other training settings unchanged. As shown in the new Table below and Figure R2.1, CFRL-SVG-2D consistently outperforms CFRL-GRR at matched total FLOPs across all four tasks, even after accounting for the full critic cost. For example, at approximately 200 PFLOPs on Countdown, SVG-2D achieves 49.6% accuracy versus 43.0% for GRR; at approximately 100 PFLOPs on Game24, SVG-2D achieves 56.4% versus 46.3% for GRR with the deployment sampler. Similar improvements hold on Sudoku and K&K.

Test accuracy (%) and total training compute (PFLOPs). The GRR deployment-sampler row is newly evaluated; all other accuracies are from Table 1. Critic-stage costs are included in the SVG totals.

| Method | Countdown Acc. | PFLOPs | Sudoku Acc. | PFLOPs | K&K Acc. | PFLOPs | Game24 Acc. | PFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Task-SFT base | 42.8 | 0 | 64.0 | 0 | 68.2 | 0 | 44.0 | 0 |
| CFRL-GRR (γ4@16) | 42.6 | 97 | 67.9 | 221 | 71.4 | 130 | 47.4 | 97 |
| CFRL-GRR (γ56@64; γ28@32 Sudoku), new | 42.8 | 208 | 64.2 | 306 | 77.0 | 278 | 47.3 | 208 |
| CFRL-SVG-1D | 47.4 | 263 | 64.1 | 269 | 76.5 | 277 | 51.2 | 232 |
| **CFRL-SVG-2D** | **49.7** | 263 | **68.7** | 269 | **77.5** | 277 | **58.2** | 232 |
| Critic stage (included in SVG) | – | 67 | – | 113 | – | 103 | – | 36 |

Our existing same-rollout experiments of Table 1 (right panels) demonstrate that the pathwise estimator provides more effective updates under identical rollout conditions. The new compute-matched experiments further show that this advantage persists after accounting for the critic's cost and matching the rollout sampler. Together, these results support that the performance gains arise from the improved optimization signal rather than simply greater computational expenditure. We have incorporated the detailed FLOPs accounting into Appendix D and added the new GRR experiments and accuracy-versus-compute curves to Section 5 of the revised manuscript, with all changes highlighted in blue.

> The unusually weak LLaDA baseline. In Table 1, after task-specific SFT, LLaDA-8B achieves only 12.1/0.5/61.4/15.9 on Countdown/Sudoku/K&K/Game24, compared with 42.8/64.0/68.2/44.0 for ELF-B with a 105M-parameter denoiser or GPT-2 performance. Is the LLaDA baseline fairly trained and evaluated? I think if the SFT/RL is well-tuned, LLaDA should have better performance and can be viewed as somewhat a upper bound given the training data.

We thank the reviewer for raising this concern. Our LLaDA baseline follows d1's released code and hyperparameters (LoRA rank 128, 6 generations per prompt, 256-token completions, 128 diffusion steps, learning rate 3×10⁻⁶, and KL weight 0.04). We use completion-only masked SFT, followed by diffu-GRPO for 8,000 inner steps (~32k rollouts), matching the rollout budget of our pathwise methods. All models use the same exact-success metric and validation-based checkpoint selection. The relatively low LLaDA accuracy reflects the greater difficulty of our benchmarks compared with d1's original tasks: Countdown uses 4–5 numbers instead of 3, Sudoku uses 9×9 grids instead of 4×4, and solutions require programmatically verified derivations. For reference, d1 reports 42.2% on 3-number Countdown and 22.1% on 4×4 Sudoku with LLaDA-8B after SFT and RL. We do not claim that these results represent the performance ceiling of masked diffusion models.

**New experiment on d1's benchmarks.** To further address the reviewer's concern, we evaluated CFRL on d1's original Countdown-3 and 4×4 Sudoku test sets (256 problems each), using d1's own reported LLaDA results as references. ELF-B was fine-tuned on a limited subset of solver-generated training examples (20k Countdown-3 examples and 200 unique Sudoku puzzles) to avoid saturating the SFT baseline. With the full training data, SFT alone achieves 88.3% and 100%, respectively. We then applied the same critic pre-training and RL procedures as in Table 1. Checkpoints were selected using separate validation data, and accuracies were averaged over 5 sampler seeds.

Test accuracy (%) on d1's original benchmarks.

| Method | 4×4 Sudoku | Countdown (3 numbers) |
|---|---:|---:|
| LLaDA-8B-Instruct (d1) | 11.7 | 20.7 |
| d1-LLaDA: SFT + diffu-GRPO (d1) | 22.1 | 42.2 |
| ELF-B task-SFT base (105M, data-limited) | 49.2 | 41.6 |
| + CFRL-GRR | 58.9 | 40.8 |
| + CFRL-SVG-1D | 95.9 | [pending] |
| + CFRL-SVG-2D | **96.1** | [pending] |

On 4×4 Sudoku, CFRL-SVG improves the ELF-B baseline from 49.2% to approximately 96%, compared with 58.9% for CFRL-GRR and 22.1% for d1-LLaDA. On Countdown-3, ELF-B starts at a comparable accuracy to d1-LLaDA (41.6% vs. 42.2%), allowing us to evaluate the effectiveness of subsequent RL improvements.

We acknowledge that the SFT supervision differs: ELF-B uses solver-generated derivations, while d1 uses s1K. Therefore, this comparison is intended to evaluate CFRL's RL improvements on an external benchmark, rather than establish an architectural upper bound or a strictly controlled comparison between ELF-B and LLaDA.

We have added these experiments and clarifications to Appendix D of the revised manuscript, with changes highlighted in blue. We thank the reviewer again for their constructive feedback. The additional compute accounting and external benchmark evaluation strengthen our empirical analysis and clarify the scope of our conclusions.

[1] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al. Training Compute-Optimal Large Language Models. NeurIPS 2022.

## Reviewer 6mKE

We thank the reviewer for their thoughtful and constructive feedback and for recognizing the strengths of CFRL, particularly its ability to enable single-transition policy updates without backpropagating through the full denoising trajectory. We also appreciate the reviewer's positive assessment of our experimental analysis, including the single-step tests, critic swaps, and frozen-critic experiments. We acknowledge the concerns regarding the consistency of our method specification, gradient attribution, and verifier implementation, and address each point below.

> Eq. (7) maximizes $V_{\phi}$, while Table 3 uses $\mathrm{logit}(V_{\phi})$. Eq. (10) sums position logits, whereas Appendix B.4 assumes mean pooling. These discrepancies affect the method's specification and theoretical justification.

We thank the reviewer for raising this question and we apologize for the confusion. For the actor objective, eq7 is the original objective and we use $\mathrm{logit}(V_{\phi})$ in SVG implementation as a surrogate for all reported results. We will clarify this in our revised manuscript. The two objectives give the same per-sample gradient direction, since $\nabla_\theta\mathrm{logit}(V_\phi)=\nabla_\theta V_\phi/\big(V_\phi(1-V_\phi)\big)$. However, directly maximizing $V_\phi$ introduces a multiplicative factor $V_\phi(1-V_\phi)$ in the pathwise gradient, which attenuates the learning signal when the critic confidently predicts success or failure. The logit transformation removes this sigmoid-induced attenuation while preserving the per-sample gradient direction, providing a stronger optimization signal, particularly for samples predicted to fail with high confidence. Table 3 confirms the effect: ascending $V_\phi$ instead lowers accuracy by 4.0 points on Game24 and 6.1 on Countdown.

Training a scalar score through classification-based supervision and subsequently optimizing the score directly is established practice. In RLHF, reward models are trained using a log-sigmoid pairwise preference loss, while the policy directly maximizes the learned scalar reward ([1], Eqs. 1–2). Similarly, hinge-loss GANs train discriminators using a margin-based classification loss, while generators directly maximize the discriminator's unbounded score ([2]). Although these approaches differ from our success-probability critic, they provide precedent for optimizing learned scalar scores rather than bounded classification probabilities. We add the clarification of these connections in the revised manuscript.

For the pooling method over positions, we used mean pooling in the implementation, consistent with 1D critic as well as Appendix B.4. The sum in Eq. (10) is a typo and we have corrected Eq. (10) highlighted in blue in the revised manuscript.

> The pathwise update includes a self-conditioning gradient absent from the REINFORCE baseline (Appendix B.3). Together with possible critic bias, this prevents attributing the gains to variance reduction alone. How much of the pathwise improvement remains when the self-conditioning gradient is detached?

We thank the reviewer for this insightful observation. We acknowledge that GRR retains the likelihood-ratio gradient through individual stochastic transitions but does not account for all gradient paths through deterministic self-conditioning. We clarify this distinction below and show how the omitted contribution can, in principle, be recovered by a full-trajectory likelihood-ratio estimator.

**1. Gradient decomposition and connection to SVG.**

Although GRR uses a binary verifier only at the final generation step, we can define the expected terminal reward conditioned on an intermediate state as $R_k(s_k)=\mathbb E[R(s_{K+1})\mid s_k]$, where $s_k=(z_k,x_{k-1})$ and $x_{k-1}=d_\theta(s_{k-1})$. Unlike the binary verifier, this conditional expected reward may be differentiable with respect to its continuous state inputs.
Following the stochastic computation graph framework ([3]), consider a fixed differentiable conditional value function $R_k$ and hold $s_{k-1}$ fixed. The local objective is $J_k(\theta)=\mathbb E_{z_k\sim\pi_\theta(\cdot\mid s_{k-1})}[R_k(z_k,d_\theta(s_{k-1}))]$. Differentiating under the integral gives

$$
\begin{aligned}
\nabla_\theta J_k
={}\underbrace{
\mathbb E\left[
R_k(s_k)\nabla_\theta
\log\pi_\theta(z_k\mid s_{k-1})
\right]
}_{\text{REINFORCE term}}+
\underbrace{
\mathbb E\left[
(\nabla_\theta d_\theta(s_{k-1}))^\top
\nabla_{x_{k-1}}R_k(s_k)
\right]
}_{\text{Self-conditioning term}}.
\end{aligned}
$$

Here, $z_k$ is held fixed when evaluating the second term. By Gaussian integration by parts, the first term is equivalent in expectation to the reparameterization gradient through $z_k$, corresponding to the first term of SVG in Eq. (23). The second term captures the direct gradient through $x_{k-1}$, corresponding to the second term of Eq. (23). This makes explicit the gradient contribution that is not directly represented by the local GRR estimator.

**2. Recovering the full gradient through trajectory-level REINFORCE.**

Importantly, the missing contribution can also be recovered without differentiating the binary terminal verifier. Consider the complete stochastic trajectory $\mathbf z=(z_1,\ldots,z_K)$, the full trajectory likelihood is $p_\theta(\mathbf z)=\prod_k\pi_\theta(z_k\mid s_{k-1})$. Assuming the terminal reward has no additional explicit parameter dependence when $\mathbf z$ is fixed, the exact REINFORCE gradient up to the final decoding step is

$$
\mathbb E\left[R(s_{K+1})\sum_kD_\theta\log\pi_\theta(z_k\mid s_{k-1})\right],
$$

where $D_\theta$ includes gradients through the recursively generated self-conditioning variables while holding the stochastic trajectory fixed. By the chain rule,

$$
\begin{aligned}
D_\theta x_{k-1}
={}\underbrace{
\partial_\theta d_\theta(s_{k-1})
}_{Local gradient}+\underbrace{
(\partial_{x_{k-2}}d_\theta)
D_\theta x_{k-2}
}_{\text{Earlier self-conditioning path}}.
\end{aligned}
$$

The local gradient $\partial_\theta d_\theta(s_{k-1})$ is taken with the input state held fixed, including the stored self-conditioning input. Since the Gaussian transition has mean $\mu_k=\lambda z_{k-1}+\eta x_{k-1}$, we obtain

$$
\begin{aligned}
D_\theta\log\pi_\theta(z_k\mid s_{k-1})
={}\underbrace{
\frac{\eta}{\sigma^2}
(\partial_\theta d_\theta(s_{k-1}))^{\top}(z_k-\mu_k)
}_{\text{GRR gradient}}+\underbrace{
\frac{\eta}{\sigma^2}
\big((\partial_{x_{k-2}}d_\theta)\,D_\theta x_{k-2}\big)^{\top}(z_k-\mu_k)
}_{\text{Omitted trajectory gradient}}.
\end{aligned}
$$

Therefore, the exact trajectory-level gradient can be decomposed as

$$
\mathbb E[g_{\mathrm{GRR}}]
+
\mathbb E\left[
R(s_{K+1})
\sum_k
\frac{\eta}{\sigma^2}
\big((\partial_{x_{k-2}}d_\theta(s_{k-1}))\,D_\theta x_{k-2}\big)^{\top}
(z_k-\mu_k)
\right],
$$

Here, $g_{\mathrm{GRR}}$ denotes the gradient implemented in GRR, which follows the transition-level likelihood-ratio formulation used in DDPO ([4]) and Flow-GRPO ([5]), which yields an unbiased trajectory-level policy gradient for samplers without parameter-dependent deterministic self-conditioning. With self-conditioning, recovering the exact trajectory-level gradient additionally requires differentiating each transition log-density through preceding denoiser predictions, substantially increasing computational and memory costs. We therefore retain the local transition gradient and omit these cross-step dependencies as a computationally motivated approximation. Similar gradient approximations have been adopted in diffusion-model optimization: DPOK ([6], Sec. 4.1 and App. A.3) omits gradients through earlier denoising transitions for efficiency, while DRaFT-K ([7]) truncates backpropagation to the final K sampling steps. Although these methods omit different gradient contributions, they provide precedent for balancing gradient completeness against computational cost.

SVG instead estimates the additional self-conditioning gradient using a differentiable critic, avoiding the need to backpropagate through the entire sampling trajectory. Consequently, its improvement over GRR may reflect both the additional gradient contribution and differences in estimator variance. To answer the question how much of the SVG improvement comes from the additional gradient term, we have done an experiment omitting the second gradient term in eq 23:

| Task | Base | GRR | SVG-2D (Detached $\hat{x}$) | SVG-2D (Full) |
|---|---:|---:|---:|---:|
| Countdown | 41.8 | 42.6 | 42.7 | **49.2** |
| Game24 | 44.5 | 46.7 | 52.7 | **58.4** |

Detaching the self-conditioning gradient reduces SVG-2D accuracy from 49.2% to 42.7% on Countdown and from 58.4% to 52.7% on Game24. The detached variant performs comparably to GRR on Countdown (42.7% vs. 42.6%) but retains a substantial advantage on Game24 (+6.0 percentage points). 

In addition, we also add an experiment omitting the self-conditional at all for both GRR and SVG for a fair comparison:

| Task | Base | GRR | SVG-2D | 
|---|---:|---:|---:|
| Countdown | 12.7 | 13.4 | **26.9** |
| Game24 | 13.6 | 15.3 | **23.1** | 

Removing self-conditioning substantially reduces the base accuracy from 41.8% to 12.7% on Countdown and from 44.5% to 13.6% on Game24. Accordingly, SVG-2D accuracy decreases from 49.2% to 26.9% on Countdown and from 58.4% to 23.1% on Game24. GRR also experiences substantial degradation, dropping from 42.6% to 13.4% on Countdown and from 46.7% to 15.3% on Game24. Despite these reductions, **SVG-2D maintains a substantial advantage over GRR** (+13.5 and +7.8 percentage points, respectively). These results suggest that self-conditioning is important for overall model performance, while the pathwise estimator retains its effectiveness even in its absence.

> Does the Game24 verifier require every input number to be used exactly once? For example, would it reject 3*8=24 for inputs [1,2,3,8]? Appendix A only describes at-most-once use for the reused Countdown verifier.

Yes, the shared verifier for both Game24 and Countdown requires every input number to be used exactly once and the resulting expression to evaluate to the target. Thus, for inputs [1, 2, 3, 8], it rejects 3*8=24 because 1 and 2 are unused. The “at-most-once” description is incomplete. We apologize for the ambiguity and have revised Appendix A to state the full acceptance criteria explicitly.

We sincerely thank the reviewer for their time, thoughtful feedback, and constructive suggestions. We have carefully addressed the concerns regarding methodological consistency, gradient attribution, and verifier implementation, and revised the manuscript accordingly, with all changes highlighted in blue. We believe these revisions have improved the clarity, rigor, and reproducibility of our work.

[1] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. NeurIPS 2022

[2] Brock, A., Donahue, J., & Simonyan, K. Large Scale GAN Training for High Fidelity Natural Image Synthesis. ICLR 2019.

[3] John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel. Gradient Estimation Using Stochastic Computation Graphs. NeurIPS 2015

[4] Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training Diffusion Models with Reinforcement Learning. ICLR 2024

[5] Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-GRPO: Training Flow Matching Models via Online RL. NeurIPS 2025

[6] Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models. NeurIPS 2023.

[7] Clark, K., Vicol, P., Swersky, K., and Fleet, D. J. Directly Fine-Tuning Diffusion Models on Differentiable Rewards. ICLR 2024

## Reviewer oqag

We thank the reviewer for their positive assessment and thoughtful feedback. We particularly appreciate their recognition of our pathwise-gradient formulation as a meaningful contribution to continuous-flow language models, as well as their acknowledgment that our gradient reproducibility, single-step intervention, frozen-critic, and critic-swap experiments provide direct evidence supporting our methodology. We also appreciate the reviewer's constructive suggestions regarding token-level credit assignment and computational efficiency. We address these concerns below and provide additional experiments and analyses to further strengthen our empirical findings.

> The evidence does not establish that the 2D critic performs token-level credit assignment in the semantic sense claimed by the paper. Every position is trained against the same sequence-level binary label. The gain could come from gradient scaling, conditioning, or optimization dynamics rather than identifying which token caused the failure. Can the authors evaluate the 2D readings on examples with known corrupted positions and test whether changing the highest-scored positions has a larger counterfactual effect than changing other positions? This would distinguish token-level credit assignment from a position-wise parameterization of a sequence-level value.

We thank the reviewer for this insightful suggestion. We agree that supervising each token-wise critic output with the same sequence-level reward does not guarantee semantic token-level credit assignment. To investigate this, we conducted two additional experiments examining whether the 2D critic identifies incorrect token positions and whether its token-wise readings respond to targeted perturbations.

**(1) Correlation between token-wise readings and incorrect positions.** We first evaluate whether the 2D critic assigns lower readings to incorrect token positions in failed Sudoku solutions. We focus on Sudoku because each output position corresponds directly to a specific answer cell, enabling unambiguous identification of incorrect tokens. We measure the AUC for discriminating correct and incorrect positions using the critic's token-wise readings, as well as P@1, the probability that the lowest-scored position is incorrect.

| Metric | range over K steps |
|---|---|
| AUC | **0.65–0.66** |
| P@1 | **0.62–0.65** |
| Random baseline (P@1) | 0.42 |

Both metrics exceed random performance. This suggests that the 2D critic's readings contain information about which token positions are incorrect, despite being trained with sequence-level supervision.

**(2) Counterfactual token corruption.** Following the reviewer's suggestion, we conduct a controlled perturbation experiment to evaluate whether the 2D critic responds to token-level errors. Starting from a correct solution, we corrupt a single token and measure the change in each position's critic reading. We report the percentage of cases in which the corrupted position exhibits the **largest decrease in its critic reading** among all positions, compared with the random-position baseline.

| Task | Corrupted position with largest reading decrease | Random baseline |
|---|---:|---:|
| Sudoku | 11–22% | 1.2% |
| K&K | 23–30%  | 6.5% |
| Countdown | 8-61% | 3.7% |
| Game24 | 13–26% | 5.6% |

Across all four tasks, the corrupted position is substantially more likely to exhibit the largest decrease in critic reading than expected by chance. These results demonstrate that the 2D critic is sensitive to localized token perturbations, providing evidence of position-specific reward attribution despite being trained with sequence-level supervision.

**Interpretation and limitations.** We acknowledge that the 2D critic's performance gains may also arise from gradient scaling, conditioning, or optimization dynamics. While our experiments show that the critic captures position-specific information and responds to localized corruption, they do not establish that semantic token-level credit assignment is the sole source of its advantage. We have therefore refine our claims to distinguish position-sensitive reward estimation from fully interpretable token-level credit assignment.

> The empirical evidence is limited to one 105M continuous-flow model that is separately fine-tuned for four synthetic tasks with exact verifiers. This is sufficient to demonstrate that CFRL can work, but not that the approach remains stable or computationally worthwhile at language-model scale. The autoregressive rows are also not matched in pretraining, model size, or task-specific adaptation, so they should not be interpreted as evidence that continuous-flow RL is competitive with autoregressive RL.



