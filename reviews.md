# This file documents the reviewer questions and comments

## Summarization across reviewers

### Method
1. Self-conditioning gradient is absent from GRR. Did we keep self-conditioning across all 3 variants? If GRR doesn't have self-conditioning gradients, we cannot attribute the SVG gains to variance alone
2. The evidence does not establish that the 2D critic performs token-level credit assignment in the semantic sense claimed by the paper. The gain could come from gradient scaling, conditioning, or optimization dynamics rather than identifying which token caused the failure.

### Experiment specific
1. Lacks compute match within our 3 formulations (2 reviewers). Need to report total training cost including critic pretraining, fresh rollouts, and repeated critic refitting, relative to CFRL-GRR.
2. LLaDA baseline is unusually weak. Is the LLaDA baseline fairly trained and evaluated?
3. Does the Game24 verifier require every input number to be used exactly once? For example, would it reject 3*8=24 for inputs [1,2,3,8]? 
4. Experiment only on small model. Does the method remain stable or computationally worthwhile at language-model scale?
5. How much of the pathwise improvement remains when the self-conditioning gradient is detached?
6. Evaluate the 2D readings on examples with known corrupted positions and test whether changing the highest-scored positions has a larger counterfactual effect than changing other positions. This would distinguish token-level credit assignment from a position-wise parameterization of a sequence-level value.

### Writing
1. Clarification on $V_{\phi}$ (eq7) vs $logit(V_{\phi})$ (table 10), and pooling over position logits: sum of logits in eq 10 vs mean pooling in appendix B
2. The autoregressive rows are also not matched in pretraining, model size, or task-specific adaptation, so they should not be interpreted as evidence that continuous-flow RL is competitive with autoregressive RL.

## Reviewer oAZU

### Critical Weaknesses (1 weakness)

The within-ELF comparison lacks compute matching and clear attribution. Appendix D reports 4–6M rollout denoiser calls for CFRL-SVG versus 1.2M for CFRL-GRR (about 2M for SVG on Sudoku), with additional critic pre-training and online computation. The same-rollout gradient experiments support update quality but do not establish performance under equal total compute. A budget-matched comparison including critic costs is needed.

### Questions For Authors (1 question)

The unusually weak LLaDA baseline. In Table 1, after task-specific SFT, LLaDA-8B achieves only 12.1/0.5/61.4/15.9 on Countdown/Sudoku/K&K/Game24, compared with 42.8/64.0/68.2/44.0 for ELF-B with a 105M-parameter denoiser or GPT-2 performance. Is the LLaDA baseline fairly trained and evaluated? I think if the SFT/RL is well-tuned, LLaDA should have better performance and can be viewed as somewhat a upper bound given the training data.

## Reviewer 6mKE

### Critical Weaknesses (2 weaknesses)

1. Eq. (7) maximizes $V_{\phi}$, while Table 3 uses $logit(V_{\phi})$. Eq. (10) sums position logits, whereas Appendix B.4 assumes mean pooling. These discrepancies affect the method's specification and theoretical justification.

2. The pathwise update includes a self-conditioning gradient absent from the REINFORCE baseline (Appendix B.3). Together with possible critic bias, this prevents attributing the gains to variance reduction alone.

### Questions For Authors (3 questions)

1. Do Tables 1–3 use $V_{\phi}$ or $logit(V_{\phi})$ as the actor objective, and a sum or mean of position logits for the critic? Which analysis applies to the implemented version?

2. How much of the pathwise improvement remains when the self-conditioning gradient is detached?

3. Does the Game24 verifier require every input number to be used exactly once? For example, would it reject 3*8=24 for inputs [1,2,3,8]? Appendix A only describes at-most-once use for the reused Countdown verifier.

## Reviewer oqag

### Critical Weaknesses (2 weaknesses)

1. The evidence does not establish that the 2D critic performs token-level credit assignment in the semantic sense claimed by the paper. Every position is trained against the same sequence-level binary label. The critic swap convincingly shows that the 2D head changes joint training, but it does not identify why. In fact, the fixed-policy probes show similar reward discrimination and highly aligned update directions for the 1D and 2D critics. The gain could therefore come from gradient scaling, conditioning, or optimization dynamics rather than identifying which token caused the failure.
    
2. The empirical evidence is limited to one 105M continuous-flow model that is separately fine-tuned for four synthetic tasks with exact verifiers. This is sufficient to demonstrate that CFRL can work, but not that the approach remains stable or computationally worthwhile at language-model scale. The autoregressive rows are also not matched in pretraining, model size, or task-specific adaptation, so they should not be interpreted as evidence that continuous-flow RL is competitive with autoregressive RL.

### Questions For Authors (2 questions)

1. Can the authors evaluate the 2D readings on examples with known corrupted positions and test whether changing the highest-scored positions has a larger counterfactual effect than changing other positions? This would distinguish token-level credit assignment from a position-wise parameterization of a sequence-level value.
    
2. What is the total training cost of CFRL, including critic pretraining, fresh rollouts, and repeated critic refitting, relative to CFRL-GRR? A compute-matched comparison would clarify whether the improvement comes from the estimator or from substantially more learned supervision.

