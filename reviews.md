# This file documents the reviewer questions and comments

## Summarization across reviewers

### Method
1. Self-conditioning gradient is absent from GRR. Since GRR doesn't have self-conditioning gradients, we cannot attribute the SVG gains to variance alone

2. The evidence does not establish that the 2D critic performs token-level credit assignment in the semantic sense claimed by the paper. The gain could come from gradient scaling, conditioning, or optimization dynamics rather than identifying which token caused the failure.

### Experiment specific
1. Lacks compute match within our 3 formulations (2 reviewers). Need to report total training cost including critic pretraining, fresh rollouts, and repeated critic refitting, relative to CFRL-GRR.

2. LLaDA baseline is unusually weak. Is the LLaDA baseline fairly trained and evaluated?


3. Does the Game24 verifier require every input number to be used exactly once? For example, would it reject 3*8=24 for inputs [1,2,3,8]? 


4. Experiment only on small model. Does the method remain 

5. How much of the pathwise improvement remains when the self-conditioning gradient is detached?

6. Evaluate the 2D readings on examples with known corrupted positions and test whether changing the highest-scored positions has a larger counterfactual effect than changing other positions. This would distinguish token-level credit assignment from a position-wise parameterization of a sequence-level value.

### Writing
1. Clarification on $V_{\phi}$ (eq7) vs $logit(V_{\phi})$ (table 10), and pooling over position logits: sum of logits in eq 10 vs mean pooling in appendix B


2. The autoregressive rows are also not matched in pretraining, model size, or task-specific adaptation, so they should not be interpreted as evidence that continuous-flow RL is competitive with autoregressive RL.


## Reviewer oAZU

### Critical Weaknesses (1 weakness)

The within-ELF comparison lacks compute matching and clear attribution. Appendix D reports 4–6M rollout denoiser calls for CFRL-SVG versus 1.2M for CFRL-GRR (about 2M for SVG on Sudoku), with additional critic pre-training and online computation. The same-rollout gradient experiments support update quality but do not establish performance under equal total compute. A budget-matched comparison including critic costs is needed.
- Jiajing's response: we didn't include critic pretraining and refit costs for SVG. For GRR, it needs to recompute the log-density of every stored rollout (16 prompts x group size 8 = 128 rollouts per iteration) with gradients for the 8 sampled steps and the decoder step, whereas SVG only differentiates one step per rollout.
- Cost accounting derived from the release settings (scripts/task_settings.sh, train_grr.sh, train_svg.sh, pretrain_critic.sh). "Forward-only" = no-grad denoiser/critic passes; "with backward" = passes that are backpropagated; "verified rollouts" = rollouts run to the end and scored by the verifier (labels consumed). SVG actor rollouts are counted only up to the sampled step k-1 (mean k = 24 of 64; 12 of 32 on Sudoku), so the RL column is smaller than the 4-6M upper bound in Appendix D. GPU-hours still to be measured on one GPU type.

  Per iteration:

  | | CFRL-GRR (gamma4@16, 600 it) | CFRL-SVG (deployment sampler, 1000-1500 it) |
  |---|---|---|
  | verified full rollouts | 128 (16 prompts x G=8) | 32 (buffer) |
  | forward-only denoiser calls | 2,048 (128 x 16) | ~2,750 (32 x 64 buffer + 32 x ~22 actor) |
  | denoiser calls with backward | 1,280 (128 x 8 transitions + 128 x 2 decoder) | 32 |
  | critic passes | 0 | ~154 forward (refit 2 x 64, AUC check) + 32 with backward |

  Per run, Countdown:

  | | CFRL-GRR (Table 1) | CFRL-GRR, deployment sampler (Appendix E control) | SVG-2D, RL | SVG-2D, critic pre-training (64k rollouts x 16 steps; 60k x 12 epochs) | SVG-2D, total |
  |---|---|---|---|---|---|
  | forward-only passes | 1.2M | 4.9M | 4.4M | 1.0M | 5.4M |
  | passes with backward | 0.77M | 0.77M | 0.10M | 0.72M | 0.82M |
  | verified rollouts | 77k | 77k | 48k | 64k | 112k |
  | GPU-hours (single GPU type) | TBD | TBD | TBD | TBD | TBD |

  Other tasks, SVG-2D totals incl. critic (forward / backward / verified rollouts): Game24 5.4M / 0.17M / 64k; K&K 5.0M / 0.34M / 64k; Sudoku 2.5M / 0.40M / 64k. CFRL-GRR is 1.2M / 0.77M / 77k on every task. Counting a backward pass as ~3 forward passes (to be measured), SVG-2D with critic costs ~2.2x GRR on Countdown, ~1.7x on Game24 and K&K, ~1.0x on Sudoku. The deployment-sampler GRR control (4.9M fwd + 0.77M bwd) already matches SVG-2D's Countdown total and exceeds the Game24 total; it gained +0.1 (Countdown) and +3.5 (Game24). To do: extend it to Sudoku and K&K; add a cheap-sampler GRR run with iterations extended to the matched budget; measure GPU-hours on one GPU type.

### Questions For Authors (1 question)

The unusually weak LLaDA baseline. In Table 1, after task-specific SFT, LLaDA-8B achieves only 12.1/0.5/61.4/15.9 on Countdown/Sudoku/K&K/Game24, compared with 42.8/64.0/68.2/44.0 for ELF-B with a 105M-parameter denoiser or GPT-2 performance. Is the LLaDA baseline fairly trained and evaluated? I think if the SFT/RL is well-tuned, LLaDA should have better performance and can be viewed as somewhat a upper bound given the training data.
 - Jiajing's response: do we need to run LLaDA's Counddown (3 numbers) and Sudoku (4*4) to see if we can match the paper's run? if so, then our run is reasonable, meaning they can't perform well on hard tasks.


## Reviewer 6mKE

### Critical Weaknesses (2 weaknesses)

1. Eq. (7) maximizes $V_{\phi}$, while Table 3 uses $logit(V_{\phi})$. Eq. (10) sums position logits, whereas Appendix B.4 assumes mean pooling. These discrepancies affect the method's specification and theoretical justification.
 - Jiajing's response: The eq6 and eq11 critic loss is correct, but eq7 actor loss is wrong. Our implementation is doing $logit(V_{\phi})$ in eq 7, so table 10 is right and we need to update eq 7. In terms of the pooling, eq 10 has a typo and it should be mean instead of sum.

2. The pathwise update includes a self-conditioning gradient absent from the REINFORCE baseline (Appendix B.3). Together with possible critic bias, this prevents attributing the gains to variance reduction alone.
 - Jiajing's response: GRR transition is Gaussian density transition and gradient can only be taken on z part with has Gaussian density, and the self-conditioning part is a deterministic call, so it doesn't have density and the gradient cannot be computed. The reviewer is right and we have to make clear that pathwise estimator made it possible for self-conditioning to take gradient, which is separate advantage in additional to variance reduction.

### Questions For Authors (3 questions)

1. Do Tables 1–3 use $V_{\phi}$ or $logit(V_{\phi})$ as the actor objective, and a sum or mean of position logits for the critic? Which analysis applies to the implemented version?

2. How much of the pathwise improvement remains when the self-conditioning gradient is detached?

3. Does the Game24 verifier require every input number to be used exactly once? For example, would it reject 3*8=24 for inputs [1,2,3,8]? Appendix A only describes at-most-once use for the reused Countdown verifier.
 - Jiajing's answer: The verifier require every number to be used exactly once. it rejects 3*8=24 for inputs [1,2,3,8] and only accepts equations that consume all four numbers.

## Reviewer oqag

### Critical Weaknesses (2 weaknesses)

1. The evidence does not establish that the 2D critic performs token-level credit assignment in the semantic sense claimed by the paper. Every position is trained against the same sequence-level binary label. The critic swap convincingly shows that the 2D head changes joint training, but it does not identify why. In fact, the fixed-policy probes show similar reward discrimination and highly aligned update directions for the 1D and 2D critics. The gain could therefore come from gradient scaling, conditioning, or optimization dynamics rather than identifying which token caused the failure.
    
2. The empirical evidence is limited to one 105M continuous-flow model that is separately fine-tuned for four synthetic tasks with exact verifiers. This is sufficient to demonstrate that CFRL can work, but not that the approach remains stable or computationally worthwhile at language-model scale. The autoregressive rows are also not matched in pretraining, model size, or task-specific adaptation, so they should not be interpreted as evidence that continuous-flow RL is competitive with autoregressive RL.
 - Xu has done the L model run.
 - Jiajing's response: Add a sentence that the GPT-2 rows are an estimator control, not a competitiveness claim.

### Questions For Authors (2 questions)

1. Can the authors evaluate the 2D readings on examples with known corrupted positions and test whether changing the highest-scored positions has a larger counterfactual effect than changing other positions? This would distinguish token-level credit assignment from a position-wise parameterization of a sequence-level value.
    
2. What is the total training cost of CFRL, including critic pretraining, fresh rollouts, and repeated critic refitting, relative to CFRL-GRR? A compute-matched comparison would clarify whether the improvement comes from the estimator or from substantially more learned supervision.

