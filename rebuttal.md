## Reviewer oAZU

We thank the reviewer for the positive assessment of our work and for recognizing the strengths of CFRL, particularly its application of pathwise policy optimization to continuous language generation and its ability to accommodate non-differentiable decoding and verification. We also appreciate the reviewer's acknowledgment that our gradient agreement and single-update experiments provide evidence of the critic's ability to guide effective policy optimization. We address the two concerns below.

> The within-ELF comparison lacks compute matching and clear attribution. Appendix D reports 4–6M rollout denoiser calls for CFRL-SVG versus 1.2M for CFRL-GRR (about 2M for SVG on Sudoku), with additional critic pre-training and online computation. The same-rollout gradient experiments support update quality but do not establish performance under equal total compute. A budget-matched comparison including critic costs is needed.

**Compute accounting.** Appendix D previously reported only rollout denoiser calls, which did not fully capture the computational costs. We have now added a comprehensive FLOPs accounting and compute-matched comparisons that include critic pre-training and online computation. We measure total training FLOPs across all network evaluations, including rollouts, actor updates, critic pre-training, and online critic updates. Following the standard parameter-based FLOPs approximation ([1]), we count a forward pass as one unit and a full backward pass as two units. We additionally approximate an input-gradient-only backward pass as one unit, since the critic backbone is frozen during RL. The full-run Table1 compute of CFRL-SVG is 1.2–2.7× that of the original CFRL-GRR, rather than the 3–5× suggested by rollout counts alone. The critic stage accounts for 14–42% of the total SVG compute.

**Compute-matched comparison.** We additionally trained CFRL-GRR using the same deployment sampler as CFRL-SVG (64 steps, or 32 for Sudoku), rather than its original 16-step sampler. This approximately doubles GRR's computational cost while keeping all other training settings unchanged. As shown in the new Table below (new table 10 in the paper), CFRL-SVG-2D consistently outperforms CFRL-GRR at matched total FLOPs across all four tasks, even after accounting for the full critic cost. And we also added figure 3 in the paper to show Acc vs Flops across four datasets on three variants of CFRL. It also shows that with the comparable compute, CFRL-SVG outperforms CFRL-GRR consistently. For example, at approximately 200 PFLOPs on Countdown, SVG-2D achieves 49.6% accuracy versus 43.0% for GRR; at approximately 100 PFLOPs on Game24, SVG-2D achieves 56.4% versus 46.3% for GRR with the deployment sampler. Similar improvements hold on Sudoku and K&K.

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

We thank the reviewer for raising this concern. The relatively low LLaDA accuracy reflects the greater difficulty of our benchmarks compared with d1's original tasks: Countdown uses 4–5 numbers instead of 3, Sudoku uses 9×9 grids instead of 4×4, and solutions require programmatically verified derivations. For reference, d1 reports 42.2% on 3-number Countdown and 22.1% on 4×4 Sudoku with LLaDA-8B after SFT and RL. We do not claim that these results represent the performance ceiling of masked diffusion models, nor do we claim continuous flow models perform better than masked diffusion models. To further address the reviewer's concern, we evaluated CFRL on d1's original Countdown-3 and 4×4 Sudoku test sets (256 problems each), using d1's own reported LLaDA results as references. ELF-B was fine-tuned on a limited subset of solver-generated training examples (20k Countdown-3 examples and 200 unique Sudoku puzzles) to avoid saturating the SFT baseline. With the full training data, SFT alone achieves 88.3% and 100%, respectively. We then applied the same critic pre-training and RL procedures as in Table 1. Checkpoints were selected using separate validation data, and accuracies were averaged over 5 sampler seeds.

Test accuracy (%) on d1's original benchmarks.

| Method | 4×4 Sudoku | Countdown (3 numbers) |
|---|---:|---:|
| LLaDA-8B-Instruct (d1) | 11.7 | 20.7 |
| d1-LLaDA: SFT + diffu-GRPO (d1) | 22.1 | 42.2 |
| ELF-B task-SFT base (105M, data-limited) | 49.2 | 41.6 |
| + CFRL-GRR | 58.9 | 43.7 |
| + CFRL-SVG-1D | 95.9 | 61.4 |
| + CFRL-SVG-2D | **96.1** | 62.8 |

On 4×4 Sudoku, CFRL-SVG and CFRL-GRR improves the ELF-B baseline from 49.2% to approximately 96% and 59% respectively, compared to d1-LLaDA that improves the baseline from 11.7% to 22.1%. On Countdown-3, CFRL-SVG and CFRL-GRR improves the ELF-B baseline from 41.6% to approximately 63% and 43.7% respectively, compared to d1-LLaDA that improves the baseline from 20.7% to 42.2%.

Our LLaDA baseline in Table 1 follows d1's released code and hyperparameters (LoRA rank 128, 6 generations per prompt, 256-token completions, 128 diffusion steps, learning rate 3×10⁻⁶, and KL weight 0.04). We use completion-only masked SFT, followed by diffu-GRPO for 8,000 inner steps (~32k rollouts), matching the rollout budget of our pathwise methods. All models use the same exact-success metric and validation-based checkpoint selection. 

We have added these experiments and clarifications to Appendix D of the revised manuscript, with changes highlighted in blue. We thank the reviewer again for their constructive feedback. The additional compute accounting and external benchmark evaluation strengthen our empirical analysis and clarify the scope of our conclusions.

[1] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al. Training Compute-Optimal Large Language Models. NeurIPS 2022.

## Reviewer 6mKE

We thank the reviewer for their thoughtful and constructive feedback and for recognizing the strengths of CFRL, particularly its ability to enable single-transition policy updates without backpropagating through the full denoising trajectory. We also appreciate the reviewer's positive assessment of our experimental analysis, including the single-step tests, critic swaps, and frozen-critic experiments. We acknowledge the concerns regarding the consistency of our method specification, gradient attribution, and verifier implementation, and address each point below.

> Eq. (7) maximizes $V_{\phi}$, while Table 3 uses $\mathrm{logit}(V_{\phi})$. Eq. (10) sums position logits, whereas Appendix B.4 assumes mean pooling. These discrepancies affect the method's specification and theoretical justification.

We thank the reviewer for raising this question. In the actor's training objective, Eq. (7) maximizes the expected reward $\mathbb{E}[V_\phi]$. Given that $V_\phi(s_k)\approx P(R=1\mid s_k)$, the actor is trained to maximize the expected reward. In our SVG implementation, we use $\mathbb{E}[\mathrm{logit}(V_{\phi})]$ instead. This design encourages the actor to stay away from wrong answers, even when its output is correct. It shares the same spirit as MaxRL [].

Maximizing this score encourages the generator toward states that are relatively more characteristic of successful than unsuccessful trajectories. Furthermore, the two objectives give the same per-sample gradient direction, since $\nabla_\theta\mathrm{logit}(V_\phi)=\nabla_\theta V_\phi/\big(V_\phi(1-V_\phi)\big)$. However, directly maximizing $V_\phi$ introduces a multiplicative factor $V_\phi(1-V_\phi)$ in the pathwise gradient, which attenuates the learning signal when the critic confidently predicts success or failure. The logit transformation removes this sigmoid-induced attenuation while preserving the per-sample gradient direction, providing a stronger optimization signal, particularly for samples predicted to fail with high confidence. Table 3 confirms the effect: ascending $V_\phi$ instead lowers accuracy.

Training a scalar score through classification-based supervision and subsequently optimizing the score directly is established practice. In RLHF, reward models are trained using a log-sigmoid pairwise preference loss, while the policy directly maximizes the learned scalar reward ([2], Eqs. 1–2). Similarly, hinge-loss GANs train discriminators using a margin-based classification loss, while generators directly maximize the discriminator's unbounded score ([3], [4]). Although these approaches differ from our success-probability critic, they provide precedent for optimizing learned scalar scores rather than bounded classification probabilities. We add the clarification of these connections in the revised manuscript.

For the pooling method over positions, we used mean pooling in the implementation, consistent with 1D critic as well as Appendix B.4. The sum in Eq. (10) is a typo and we have corrected Eq. (10) highlighted in blue in the revised manuscript.

> The pathwise update includes a self-conditioning gradient absent from the REINFORCE baseline (Appendix B.3). Together with possible critic bias, this prevents attributing the gains to variance reduction alone. How much of the pathwise improvement remains when the self-conditioning gradient is detached?

We thank the reviewer for this insightful observation. We acknowledge that GRR retains the REINFORCE gradient computed through one denoising step but does not account the gradient from the computation path of deterministic self-conditioning.

In our new version, we clarify the action definition to be $a_k = \eta_k \mathbf{x}_k + \psi_k \epsilon$ and then update our derivation in the submission as follows. The optimization of the actor's objective $\mathcal J_{\mathrm{GRR}}(\theta)$ (Eq. 5)requires the computation of the gradient $\nabla_\theta \log\pi_\theta\bigl(a^g_k \mid s^g_{k}\bigr)$ for each group $g$. Below we omit $g$ for notation simplicity. Note that $a_k$ is from a Gaussian distribution $\mathcal{N}(\eta_k \rvx_{t_{k}}, \psi_k^2 c_\sigma^2 I)$, and we have:

\begin{align*}

\nabla_\theta \log\pi_\theta\bigl(a_k \mid s_{k}\bigr) &= \frac{(a_k - \eta_k \rvx_{t_{k}})^\top }{\psi_{k}^2 c_\sigma^2} \left( \frac{\partial \rvx_{t_{k}}}{\partial \theta} + 

\frac{\partial \rvx_{t_{k}}}{\partial \rvx_{t_{k-1}}} \cdot D_{\theta} \rvx_{t_{k-1}} \right)  \approx \frac{(a_k - \eta_k \rvx_{t_{k}})^\top }{\psi_{k}^2 c_\sigma^2}\frac{\partial \rvx_{t_{k}}}{\partial \theta}.

\end{align*}

There is a determinstic computation path from $\rvx_{t_0}$ to $\rvx_{t_{k}}$ through the self-conditioning mechanism in the denoiser. The term $D_\theta \rvx_{t_{k-1}}$ is the total derivative of $\rvx_{t_{k-1}}$ with respect to $\theta$ through the path. The exact gradient calculation requires backpropagation through denoising steps, which has the same issue as backpropagation through time. In our implementation, we truncate the backpropagation through steps and only use the local gradient $\frac{\partial \rvx_{t_{k}}}{\partial \theta}$ as the approximate gradient. 

From this derivation, we agree that the gradient of GRR is biased, and computing the unbiased estimation is computationally hard. Similar gradient approximations have been adopted in diffusion-model optimization: DPOK ([7], Sec. 4.1 and App. A.3) omits gradients through earlier denoising transitions for efficiency, while DRaFT-K ([8]) truncates backpropagation to the final K sampling steps. 

Therefore, bias in gradient estimation is a weakness of GRR. In contrast, SVG estimates the additional self-conditioning gradient using a differentiable critic, avoiding the need to backpropagate through the entire sampling trajectory. Consequently, SVG's improvement over GRR may reflect its ability to reduce both bias and variance in gradient estimation. To isolate the contribution of gradient estimation to performance, we conduct two additional experiments. First, we remove the second gradient term from SVG to match the gradient formulation of GRR. Second, we remove self-conditioning, under which the GRR gradient estimator becomes unbiased.

The table below show the results from the first experiment. 

| Task | Base | GRR | SVG-2D (Detached $\hat{x}$) | SVG-2D (Full) |

|---|---:|---:|---:|---:|

| Countdown | 41.8 | 42.6 | 42.7 | **49.2** |

| Game24 | 44.5 | 46.7 | 52.7 | **58.4** |

Detaching the self-conditioning gradient reduces SVG-2D accuracy from 49.2% to 42.7% on Countdown and from 58.4% to 52.7% on Game24. The detached variant performs comparably to GRR on Countdown (42.7% vs. 42.6%) but retains a substantial advantage on Game24 (+6.0 percentage points). 

We also add an experiment omitting the self-conditional at all for both GRR and SVG for a fair comparison:

| Task | Base | GRR | SVG-2D | 

|---|---:|---:|---:|

| Countdown | 12.7 | 13.4 | **26.9** |

| Game24 | 13.6 | 15.3 | **23.1** | 

Removing self-conditioning substantially reduces the base accuracy from 41.8% to 12.7% on Countdown and from 44.5% to 13.6% on Game24. Accordingly, SVG-2D accuracy decreases from 49.2% to 26.9% on Countdown and from 58.4% to 23.1% on Game24. GRR also experiences substantial degradation, dropping from 42.6% to 13.4% on Countdown and from 46.7% to 15.3% on Game24. Despite these reductions, **SVG-2D maintains a substantial advantage over GRR** (+13.5 and +7.8 percentage points, respectively). These results suggest that self-conditioning is important for overall model performance, while the pathwise estimator retains its effectiveness even in its absence.

> Does the Game24 verifier require every input number to be used exactly once? For example, would it reject 3*8=24 for inputs [1,2,3,8]? Appendix A only describes at-most-once use for the reused Countdown verifier.

Yes, the shared verifier for both Game24 and Countdown requires every input number to be used exactly once and the resulting expression to evaluate to the target. Thus, for inputs [1, 2, 3, 8], it rejects 3*8=24 because 1 and 2 are unused. The “at-most-once” description is incomplete. We apologize for the ambiguity and have revised Appendix A to state the full acceptance criteria explicitly.

We sincerely thank the reviewer for their time, thoughtful feedback, and constructive suggestions. We have carefully addressed the concerns regarding methodological consistency, gradient attribution, and verifier implementation, and revised the manuscript accordingly, with all changes highlighted in blue. We believe these revisions have improved the clarity, rigor, and reproducibility of our work.

[1] Gutmann, M., & Hyvärinen, A. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. AISTATS 2010.

[2] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. NeurIPS 2022.

[3] Brock, A., Donahue, J., & Simonyan, K. Large Scale GAN Training for High Fidelity Natural Image Synthesis. ICLR 2019.

[4] Lim, J. H., & Ye, J. C. (2017). Geometric GAN. arXiv:1705.02894.

[5] Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training Diffusion Models with Reinforcement Learning. ICLR 2024.

[6] Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-GRPO: Training Flow Matching Models via Online RL. NeurIPS 2025.

[7] Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models. NeurIPS 2023.

[8] Clark, K., Vicol, P., Swersky, K., and Fleet, D. J. Directly Fine-Tuning Diffusion Models on Differentiable Rewards. ICLR 2024.

## Reviewer oqag

We thank the reviewer for their positive assessment and thoughtful feedback. We particularly appreciate their recognition of our pathwise-gradient formulation as a meaningful contribution to continuous-flow language models, as well as their acknowledgment that our gradient reproducibility, single-step intervention, frozen-critic, and critic-swap experiments provide direct evidence supporting our methodology. We also appreciate the reviewer's constructive suggestions regarding token-level credit assignment and computational efficiency. We address these concerns below and provide additional experiments and analyses to further strengthen our empirical findings.

> The evidence does not establish that the 2D critic performs token-level credit assignment in the semantic sense claimed by the paper. Every position is trained against the same sequence-level binary label. The gain could come from gradient scaling, conditioning, or optimization dynamics rather than identifying which token caused the failure. Can the authors evaluate the 2D readings on examples with known corrupted positions and test whether changing the highest-scored positions has a larger counterfactual effect than changing other positions? This would distinguish token-level credit assignment from a position-wise parameterization of a sequence-level value.

We thank the reviewer for this insightful suggestion. We agree that supervising each token-wise critic output with the same sequence-level reward does not guarantee semantic token-level credit assignment. To investigate this, we conducted two additional experiments examining whether the 2D critic identifies incorrect token positions and whether its token-wise readings respond to targeted perturbations.

**(1) Correlation between token-wise readings and incorrect positions.** We first evaluate whether the 2D critic assigns lower readings to incorrect token positions in failed Sudoku solutions. We focus on Sudoku because each output position corresponds directly to a specific answer cell, enabling unambiguous identification of incorrect tokens. We use precision to measure whether the lowest-scored position is incorrect.

| Metric | range over K steps |
|---|---|
| Precision | **0.62–0.65** |
| Random baseline precision | 0.42 |

The precision exceeds random performance. This suggests that the 2D critic's readings contain information about which token positions are incorrect, despite being trained with sequence-level supervision.

**(2) Counterfactual token corruption.** Following the reviewer's suggestion, we conduct a controlled perturbation experiment to evaluate whether the 2D critic responds to token-level errors. Starting from a correct solution, we corrupt a single token and measure the change in each position's critic reading. We report the percentage of cases in which the corrupted position exhibits the **largest decrease in its critic reading** among all positions, compared with the random-position baseline.

| Task | Corrupted position with largest reading decrease | Random baseline |
|---|---:|---:|
| Sudoku | 11–22% | 1.2% |
| K&K | 23–30%  | 6.5% |
| Countdown | 8-61% | 3.7% |
| Game24 | 13–26% | 5.6% |

Across all four tasks, the corrupted position is substantially more likely to exhibit the largest decrease in critic reading than expected by chance. An incorrect answer token can be sufficient to establish sequence-level failure, whereas a correct token alone cannot establish success. This asymmetry motivates position-sensitive critic readings even under sequence-level supervision. These results demonstrate that the 2D critic is sensitive to localized token perturbations, providing evidence of position-specific reward attribution despite being trained with sequence-level supervision. 

**Interpretation and limitations.**  Our experiments show that the critic captures position-specific information and responds to localized corruption, while they do not establish that semantic token-level credit assignment is the sole source of its advantage and we do not rule out the possibility that the 2D critic's performance gains may also arise from gradient scaling, conditioning, or optimization dynamics. We have therefore refined our claims to distinguish position-sensitive reward estimation from fully interpretable token-level credit assignment.

> The empirical evidence is limited to one 105M continuous-flow model that is separately fine-tuned for four synthetic tasks with exact verifiers. This is sufficient to demonstrate that CFRL can work, but not that the approach remains stable or computationally worthwhile at language-model scale. The autoregressive rows are also not matched in pretraining, model size, or task-specific adaptation, so they should not be interpreted as evidence that continuous-flow RL is competitive with autoregressive RL.

We thank the reviewer for highlighting the importance of scalability and fair comparisons. We agree that our autoregressive (AR) baselines, which differ in pretraining, model size, and task-specific adaptation, should not be interpreted as evidence that continuous-flow RL is competitive with autoregressive RL. We have clarified that these results serve only as reference points rather than controlled comparisons.

To investigate whether CFRL's improvements extend beyond the 105M-parameter ELF-B model, we conducted additional experiments on **ELF-L (652M parameters)**, approximately 6.2× larger than ELF-B. We applied the same critic pre-training and RL hyperparameters without additional tuning, using the same validation-based checkpoint selection and averaging test accuracy over five sampling seeds.

Test accuracy (%) on ELF-L (652M). Improvements over task-SFT are shown in parentheses. The final column reports the corresponding SVG-2D improvement on ELF-B (105M) for comparison.

| Task | SFT Base | CFRL-GRR | CFRL-SVG-1D | CFRL-SVG-2D | ELF-B SVG-2D Gain |
|---|---:|---:|---:|---:|---:|
| Countdown | 60.5 | 63.3 (+2.8) | 72.6 (+12.1) | **74.4 (+13.9)** | +6.9 |
| Sudoku | 71.0 | 70.3 (-0.7) | **79.3 (+8.3)** | 77.9 (+6.9) | +4.7 |
| Game24 | 58.6 | 66.9 (+8.3) | 70.5 (+11.9) | **71.0 (+12.4)** | +14.2 |
| K&K | 100.0 | – | – | – | SFT saturated |

CFRL-SVG consistently improves over the SFT baseline and CFRL-GRR across all three non-saturated tasks. In particular, SVG-2D achieves improvements of +13.9, +6.9, and +12.4 percentage points on Countdown, Sudoku, and Game24, respectively. These gains are comparable to or greater than those observed on ELF-B, demonstrating that CFRL's effectiveness extends to a substantially larger continuous-flow model. On Sudoku, SVG-1D slightly outperforms SVG-2D, suggesting that the relative benefit of the token-wise critic can vary with model size and task.

While these experiments provide encouraging evidence of **scalability from 105M to 652M parameters**, we acknowledge that they do not establish stability or computational efficiency at multi-billion-parameter scale. We have therefore clarified the scope of our claims and identified further scaling studies as future work.

> What is the total training cost of CFRL, including critic pretraining, fresh rollouts, and repeated critic refitting, relative to CFRL-GRR? A compute-matched comparison would clarify whether the improvement comes from the estimator or from substantially more learned supervision.

We thank the reviewer for raising this question about compute matching. We measure total training FLOPs across all network evaluations, including rollouts, actor updates, critic pre-training, and online critic updates. Following the standard parameter-based FLOPs approximation ([1]), we count a forward pass as one unit and a full backward pass as two units. We additionally approximate an input-gradient-only backward pass as one unit, since the critic backbone is frozen during RL. The full-run Table1 compute of CFRL-SVG is 1.2–2.7× that of the original CFRL-GRR. The critic stage accounts for 14–42% of the total SVG compute.

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

We sincerely thank the reviewer for their time and thoughtful feedback, which have helped us strengthen the paper and improve its clarity and rigor. We have incorporated these additional experiments and clarifications into the revised manuscript, with all changes highlighted in blue.

[1] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al. Training Compute-Optimal Large Language Models. NeurIPS 2022.

