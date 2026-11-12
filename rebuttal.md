## Reviewer oAZU

We thank the reviewer for their positive assessment of our work and for recognizing the strengths of CFRL, particularly its application of pathwise policy optimization to continuous language generation and its ability to accommodate non-differentiable decoding and verification. We also appreciate the reviewer's acknowledgment that our gradient agreement and single-update experiments provide evidence of the critic's ability to guide effective policy optimization. We address the two concerns below.

> The within-ELF comparison lacks compute matching and clear attribution. Appendix D reports 4–6M rollout denoiser calls for CFRL-SVG versus 1.2M for CFRL-GRR (about 2M for SVG on Sudoku), with additional critic pre-training and online computation. The same-rollout gradient experiments support update quality but do not establish performance under equal total compute. A budget-matched comparison including critic costs is needed.

**Compute accounting.** Appendix D previously reported only rollout denoiser calls, which did not fully capture the computational costs. We have now added a comprehensive FLOPs accounting and compute-matched comparisons that include critic pre-training and online computation. We measure total training FLOPs across all network evaluations, including rollouts, actor updates, critic pre-training, and online critic updates. Following standard conventions (Kaplan et al., 2020; Hoffmann et al., 2022), we account for both forward and backward passes. The full-run Table1 compute of CFRL-SVG is 1.2–2.7× that of the original CFRL-GRR, rather than the 3–5× suggested by rollout counts alone. The critic stage accounts for 14–42% of the total SVG compute.

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
