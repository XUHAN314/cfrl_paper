# 对 comment_09_23_2.md 的逐条回复（2026-09-24）

说明：逐条对应 `comment_09_23_2.md` 的编号。每条给"同意 / 不同意 / 部分同意"；同意的写 main 里怎么改，不同意的给理由。这一轮同时做了两件事：(1) 把两个已核对的实验写进论文（见文末"新加入的实验"）；(2) 对 §5 做了一遍写作质量整理（见"§5 写作整理"）。改完过两轮 subagent 审查，审查里发现的问题列在最后。

## 关于"实验部分有些地方改回去了"

git 里没有回退（coauthor 的提交只加了 comment 文件）。发生的是：上一轮按 comment_09_23_1 把 §5 压缩后，subagent 审查挑出压缩句不够准（例如 "on the other tasks the 2D critic likewise ends ahead" 会被读成高于 base），我又把句子展开回去，等于把压缩撑回去了。三处：2D-critic 段的 Countdown/K&K/Sudoku 逐个描述、主结果里 Sudoku 的 "no seed clears the base…" 半句、Ablations 段的 Countdown 半句。这轮的处理原则改为：正文压缩加指针，准确性的细节只改附录（E.3.3 已有完整描述）。这三处这次都按 comment 压回去了，并且给审查 agent 加了"正文不许展开、准确性问题改附录"的规则。

## Section 2

1. **同意。** 改为 "A call may use classifier-free guidance (Appendix D); we treat the guided evaluation as one denoiser call $\hat X_\theta$."
2. **同意。** 改为 "adds fresh noise to the state (\emph{re-noising}) before every denoiser call but the last"。
3. **不同意** Liping 把 estimate 行改成 $\hat x_{k-1}=\hat X_\theta(y_{k-1},\tilde t_{k-1};\hat x_{k-2})$。display 是按"步"排的：第 $k$ 步 = 加噪得 $y_k$、在 $s_k=(y_k,\hat x_{k-1})$ 处调用得 $\hat x_k$、partial step 得 $z_k=y_k+\lambda_k(\hat x_k-y_k)$。若 estimate 行改成 $k-1$，partial-step 行用到的 $\hat x_k$ 就没有定义。"transition $k$ = $s_{k-1}$ 处的调用 + 加噪得 $y_k$"这个按转移的分法在 display 下一段已经写明（"The display groups the operations by step; the density below groups them by transition…"）。为了帮读者，把 Markov 那句改成 "The pair $s_k=(y_k,\hat x_{k-1})$ is the complete input of the call of step $k$, whose output is $\hat x_k$"。
4. **部分同意。** 递推本身要留（§4.1 的分解和 bias 的讨论都要用"$V^\star_k$ 依赖 $\theta$"这个事实），但 "so" 的逻辑确实跳了。现在写成 "For $2\le k\le K-1$ the tower property gives $V^\star_{k-1}(s)=\dots$: each value is a Gaussian average of the next over $y_k$, and every $V^\star_k$ depends on $\theta$ through the calls made after step $k$." 评论里"过程只有 y 和 x，没有 re-noise"是误读：$y_k=\alpha_kD_\theta(s)+\sigma_k\varepsilon_k$ 这一项就是 re-noise（$z$ 以 $D_\theta(s)$ 的形式出现）。
5. **同意。** 平滑/直传那句从 §2 删掉；§2 末段（两种估计量的过渡段）里前引 §3/§4/附录 B.4 的括号也一并删掉，预备知识节不再指向后文。这个性质只在 §4.1 用到一次，放那里（见 §4 第 4 条）。

## Section 3

1. **同意。** 改为 "We recompute the transition and readout factors from the stored states $s^g_{k-1}$ and $s^g_{K-1}$, so that they depend on $\theta$ ($z^g_K=z_K(s^g_{K-1})$), and minimise"。
2. **部分同意。** score 是 $\nabla_\theta\log\pi$（score function），不是 $J$ 的梯度，也是 "score-function estimator" 名字的来源。保留 score，写出定义："is the score $\nabla_\theta\log\pi^k_\theta$ of a transition factor"。
3. **同意。** "and leaves" 改 "but leaves"。

## Section 4

1. **部分同意。** 这句 SVG 定位是 09-21 作者要求加的（点明 SVG(1) 结构）。保留一个短语 "an SVG whose transition model is the sampler itself"，删掉后半段 "and the value is the probability of success, fitted to verifier outcomes and refit as the policy moves"（后文都会讲到）。
2. **同意。** 删 "this is bookkeeping, not a copy of the parameters"。
3. **不同意** Liping 把 $=$ 改成 $\approx$。$\theta$ 在 $J$ 里出现 $K-1$ 次（每次调用一次），多元链式法则 $\frac{dJ}{d\theta}=\sum_k\frac{\partial J}{\partial(\text{第 }k\text{ 次出现})}$ 是恒等式，写 $\approx$ 是错的。近似发生在别处并已写明：只抽窗口里一个 $k$（估计量），以及用 critic 代替 $V^\star_k$（bias $b_k$）。为避免再被误读，句子改为 "since $\theta$ enters $J$ once per call, the multivariable chain rule gives $\nabla_\theta J=\sum_{k=2}^{K}\nabla^{(k)}_\theta J$ exactly"。
4. **部分同意。** 这句是 eq. (8) 成立的依据，不能删：reward 是分段常数，不能直接在期望里面求导；靠的是下一次加噪把 $V^\star_k$ 变成高斯卷积、对均值可导，这也是 $k=K-1$ 没有 pathwise 信号的原因。改写成不带 "re-noising" 的说法并挂到 §2 的递推上："For $k\le K-2$ the recursion of §2 makes $V^\star_k$ a Gaussian average of $V^\star_{k+1}$ centred at $\alpha_{k+1}D_\theta(s_k)$, hence differentiable in that mean although the reward is not; assuming the same regularity in the estimate $\hat x_{k-1}$ (Appendix B.4), the expectation is differentiated inside with the noise held fixed"。§2 里的那份已删（§2 第 5 条），全文只此一处。
5. **同意。** 改为 "The critic $V_\phi$ is trained to approximate $\mathrm{logit}\,V^\star_k$ (§4.2)."
6. **部分同意。** 用建议的结构，保留 "memory does not grow with $K$" 这个具体好处，不写 "back propagation through time issue"（太泛）："The actor gradient passes through the sampled transition only, not the whole chain: the critic carries the later transitions in expectation, so memory does not grow with $K$."
7. **同意。** 按建议改写；因为 $1/[V(1-V)]$ 这个权重在 eq. (9) 前已经出现，这里只说它的作用："Cross-entropy trains the critic's raw output to approximate the log-odds of success, and the weight $1/[V^\star_k(1-V^\star_k)]$ above amplifies the probability gradient where the outcome is nearly decided."
8. **同意。** 窗口那句并到窗口定义处。
9. **同意。** 删 baseline 那句。
10. **同意。** 改为 "The update carries the bias $b_k$ of …"。
11. **同意。** 改为 "The bias vanishes when the critic's slope matches that of the log-odds on the states the current policy visits, which motivates the critic's design (§4.2) and its training (§4.3)."
12. **同意。** 改为 "The critic takes the state $s_k=(y_k,\hat x_{k-1})$ and the time $\tilde t_k$ as input. Because $\hat x_{k-1}$ is the denoiser's current prediction of the decoded sequence, the critic reuses the denoiser to read it: a copy $\hat X_{\bar\theta}$, …"（后半句接原来的 copy 定义）。
13. **同意。** 改为 "At a given training step the two updates are similar, because both critics use the same features and outcome labels (Appendices E.3.4, E.3.5); they diverge over training as the policy evolves and the heads are refitted (Figure 2)."
14. **不同意**搬到实验节。这两句是设计选择的理由（步长衰减因为 bias 随步长增大；在部署采样器下训练因为换采样器就换了 value 函数），不是实验结果；按方法节规则它们正好属于这里。压成一句："The bias paid per step grows with the step size, hence a decaying schedule; and both networks are trained under the sampler to be deployed, since a different sampler is a different chain with different values (exception: Appendix C.2)."
15. **同意。** 改为 "The SVG variants' gains are specific to their training chain and need not transfer to a cheaper chain through the same denoiser (Appendix E.4.2)."
16. **同意。** 改为 "An optional amortisation step transfers them to other samplers through the supervised objective, …"。
17. **同意。** 改为 "A round learns only from prompts with accepted responses, so its reach is limited by the pool's coverage, which the 2D critic can help expand: …"。
18. **部分同意。** 采用建议的句子结构，"score" 一词去掉；删掉实验结果指针 "(tested in Appendix E.4.1)"，保留方法指针 Appendix C.3（那里是算法描述，不是结果）："it uses its most pessimistic position, $\min_i\mathrm{sig}(R_{k,i})$, to prune parallel chains and to guide the state before decoding, so that pools can be drawn with a cheap chain (Appendix C.3)."

## Section 5

1. **同意。** 删掉 Sudoku 的 "where no seed clears the base…" 半句，只留 "(Sudoku excepted)"；Table 1 的 +0.1 读者自己看得到。
2. **是。** 就是把 $\beta>0$ 的 KL anchor 加回来。改为 "restoring the KL anchor ($\beta>0$) removes even those points"。
3. **部分同意。** 用建议的句式，但 $\Lambda$ 的范围要留：一个 $\Lambda$ 时 pathwise 的增益反转，去掉范围这句就不真了。"At a quarter and a half of $\Lambda$ the pathwise step improves accuracy with intervals that exclude zero, whereas the score-function step gains under two points with intervals that include zero; at a full $\Lambda$ the pathwise gain turns over, …"
4. **同意。** Table 2 右图图例改 2×2，去掉 "(right axis)"；标签缩短为 batch success / held-out acc. / critic value / base 才放得下（`figures/make_compact.py`）。图注里保留 "the critic's value is on the right axis, in logits"。
5. **同意。** 删 "Game24"。
6. **同意。** 正文只留 "(other tasks: Appendix E.3.3)"。这就是上面说的"改回去"的地方，附录 E.3.3 有每个任务的描述。
7. **同意。** 改为 "On states sampled from one policy, however, the two critics perform similarly (Appendix E.3.4)."

## 新加入的实验（两个，都已核对）

**A. GPT-2 PPO 改用 dropout 关闭的 recipe，三个 seed。** 之前 Table 1 的 GPT-2 行是 dropout 开着跑的 RL（Countdown 53.9、Sudoku 1.4、Game24 32.2），K&K 的 93.2 那次实际上已经是 dropout 关闭的（09-15 的 `kk_rl_3e-6` 与 `kk_rl_nd_3e6` 曲线逐点相同），也就是原来四个任务混用了两种 recipe。dropout 关闭是 PPO 的标准做法：算 importance ratio 的两次前向必须一致，否则 ratio 带噪、clip 在参数没动时就触发。现在统一为 dropout 关闭、每任务三个 seed（seed 0 = 09-15 的 run，seed 1–2 = 09-23 的作业 4241089–4241105）：

| 任务 | SFT | 旧 Table 1（dropout on） | 新（dropout off，3 seeds） | 三个 seed |
|---|---|---|---|---|
| Countdown | 44.2 | 53.9 (+9.7) | 61.8 (+17.6) | 61.7 / 62.5 / 61.1 |
| Sudoku | 1.2 | 1.4 (+0.2) | 1.4 (+0.2) | 1.6 / 1.6 / 0.9 |
| K&K | 92.9 | 93.2 (+0.3) | 93.3 (+0.4) | 93.2 / 93.2 / 93.5 |
| Game24 | 26.9 | 32.2 (+5.3) | 46.6 (+19.7) | 46.2 / 47.3 / 46.2 |

dropout 开着的对照 seed（54.1/54.1、29.6/27.4；K&K 上 3e-6 也塌到 0）说明差别不是 seed 噪声。**对结论的影响**：不破坏任何主张，反而更强——"同一个 score-function 估计量在 token 上不封顶"现在在 Countdown 和 Game24 两个任务上都成立（之前 Game24 上 GPT-2 只有 +5，与两步 GRPO 变体打平，这个说法被限制在 Countdown）。绝对水平上 GPT-2 在 Countdown 领先 ELF-B 2D 的幅度从 4 点变成 12 点；论文没有"打败 AR"的主张，GPT-2 只是 cross-architecture reference。**改动的位置**：Table 1 GPT-2 行和图注（GPT-2 也是三 seed 均值）；§5 主结果（第一版写 "about eighteen points on Countdown and twenty on Game24"，第一轮审查后数字删去，改为 "an order of magnitude more"）；Appendix D 的 AR control 段（写明 dropout 关闭及原因、三 seed 峰值）和 seeds 段；Appendix E.1.2 的文字和曲线表（下半部分改为 dropout off 的曲线 + dropout on 的参考行，K&K 给一条塌掉的 dropout-on 曲线）；E.2.1 表里 GPT-2 那行 +9.7 → +17.6（表注和正文里的 +9.7 同步）；附录结果图的虚线（`figures/make_results.py`）。

**B. GPT-2 在 Game24 上的梯度 half-split cosine（作业 4246904）。** 之前 Table 1 右上面板里 GPT-2 那条线是 Countdown 上测的（其余三条是 Game24），coauthor 要求面板单数据集。用同一脚本、同样的 prompt 数和 seed 在 Game24 上重测：

| N | GPT-2 GRPO 梯度 | GPT-2 自己的 A=1 控制 | （Countdown 旧值） |
|---|---|---|---|
| 256 | 0.09 ± 0.05 | 0.08 ± 0.05 | 0.05 |
| 1024 | 0.35 ± 0.04 | 0.00 ± 0.06 | 0.17 |
| 2048 | 0.34 ± 0.07 | 0.05 ± 0.08 | 0.27 |

方向一致（随 N 增大离开控制线，CFRL-GRPO 停在 0.04/0.02/−0.04，pathwise 0.79/0.94/0.97）。两个如实写进附录的细节：N=256 时还没和自己的控制线分开，分开从 N=1024 起；1024→2048 是平台（误差内）。**改动的位置**：面板换成 Game24 线，图例标 "GPT-2"，图注 "Game24 rollouts of the base"；§5 删 "(measured on Countdown)"；Appendix E.2.3 的表加 Game24 行（Countdown 行保留，表注同步）和控制线数值；E.2.5 的表加 Game24 六行（GRPO 与控制 × 三个 N），文字改成 "cosine 0.00–0.02 on Countdown, 0.00–0.08 on Game24 … on Game24 to a plateau of about 0.35 from N=1024"。

## §5 写作整理（针对"实验部分写作质量不高"）

- 主结果段：先按"三个比较"重写，第一轮审查后再改为 "Read Table 1 down the ELF-B block, then across models" 的结构；正文不再重复表里的任何数字（GPT-2 对比写成 "an order of magnitude more than CFRL-GRPO wherever the supervised model leaves headroom"）。
- 2D-critic 段：Game24 图讲清一件事（scalar 平、2D 继续升），其他任务指针；"interchangeable" 改为 "perform similarly"；段落顺序是 看图 → 但静态上一样 → 交换实验定位差异 → 结论。
- Ablations 段只讲 Game24：Countdown 半句从 §5 删掉（refit 在 $k_{\rm hi}$ 的理由留在 E.3.10），refit 那句标 "(not tabulated)"，因为 Table 3 已没有这一行。
- "Why capped" 段：KL anchor 写明、Λ 那句按建议句式、去掉 Countdown 标注。

## 审查后额外修改的地方

两轮 subagent 审查（每轮两位：一位审 §2–§5 与附录 B/C/D 的数学、数字一致性和 §5 写作；一位审本 MD 与 tex 是否一致）。审查 prompt 里写明：正文不许展开，准确性问题改附录。

**第一轮（21 条 + 10 条）：**
- §2 值函数段："depends on $\theta$ through the calls made after step $k$" 不准：$V^\star_k(s_k)$ 读的是 $D_\theta(s_k)$ 和 $\hat X_\theta(s_k)$，即第 $k$ 步那次调用本身；改为 "only through the calls at $s_k$ and later (transitions $k{+}1,\dots,K$)"。§2 末段（过渡段）里前引 §3/§4/附录的括号删掉。
- §4.1 正则性假设的对象写错：没被高斯平均的是 $V^\star_{k+1}$ 的第二个参数 $\hat x_k=\hat X_\theta(s_k)$（它同时依赖 $y_k$ 和 $\hat x_{k-1}$），不是 $\hat x_{k-1}$；改为 "assuming the same regularity in the estimate $\hat x_k=\hat X_\theta(s_k)$ that $V^\star_{k+1}$ reads"，与附录 B.4 一致。eq. (9) 后 "Cross-entropy trains the critic's raw output…" 与前一句重复，删；"The update carries the bias $b_k$ of [公式]" 不通，改为 "With a learned critic the update is biased, $\mathbb E[\hat g_{\rm PW}\mid k]=\dots$"。
- 附录 B.4 开头 "transitions after step $k$ … and through no transition" 混用 step/transition 语言，改为 "through transitions $k{+}1,\dots,K$; for $V^\star_{K-1}$ that is transition $K$ alone … and no Gaussian factor of (2)"。
- §5 主结果段再改一版（审查意见：四个数字表里都有、"three comparisons" 里 LLaDA 没有结论）："Read Table 1 down the ELF-B block, then across models. … The cap is not intrinsic to the score-function estimator: GPT-2's token-level GRPO/PPO gains an order of magnitude more than CFRL-GRPO wherever the supervised model leaves headroom, while LLaDA's masked-diffusion surrogate gains a few points at most"。"The statistic measures variance, not bias." 孤句改为 "Agreement is not correctness." 接 Bottom 面板；"randomly initialised copy" 改 "network"，"updating the copy rather than the head" 与表格行 "copy also trained in RL" 意思相反，改为 "training the copy along with the head"；"Neither critic remains effective when frozen, so the advantage arises from how these choices interact with the refit" 的 "so" 过强，改为 "Both critics are exploited when frozen, so the advantage can only arise under the refit"；Ablations 段：Table 3 已没有 refit-step 行，那句挪到步长句之后、last-row 句之前，并标 "(not tabulated)"，去掉重复的 "Here" 和与 critic 段重复的 "as the bias leads one to expect"。Table 1 图注写明右图是 "each model's SFT base" 且 control 上图是 $A{=}1$、下图是随机方向。Setup 里 LLaDA 的引用改为模型 (nie2025llada) + recipe (zhao2025d1)。GPT-2 方法名统一为 "token-level GRPO/PPO"（§5、Appendix D 两处、E.1.2 正文与表注；结果图标签 "GPT-2 + GRPO/PPO"）。
- 附录 D/E：E.2.1 正文和表注里的 "+9.7" 改 "+17.6 (three-seed mean)"（上一版只改了表行）；E.1.2 "after an initial dip" 是 dropout-on 曲线的特征，dropout-off 曲线没有 dip，改为 "much of it within the first 100 iterations"；dropout-on 的对照改成三 seed 均值 "+9.8 / +2.8 (three seeds: 53.9/54.1/54.1 and 32.2/29.6/27.4)"；K&K 1e-5 那条注明 dropout off、未列表；末句 "the only difference is the estimator its action space admits" 与 "same estimator" 矛盾，改 "what differs is the action space the estimator acts on"；表注 "best RL cell" 改 "best cell of each row (earliest on ties)"，Sudoku 行加粗改到最早的并列格。E.2.3 表注改 "Last two rows … Game24 (the panel's line) and Countdown"。E.2.5 首句语病修正；"plateau … as a real signal does" 自相矛盾（常数信号 $s\approx0.03$ 预测 2048 时约 0.5），改为 "grows with N on Countdown as a real signal does (on Game24 it reaches about 0.35 at N=1024 and stays there at 2048, below the ≈0.5 a constant s≈0.03 would predict)"。
- 本 MD：三处附录编号改正（C.2、E.2.1、E.2.3）；"两个数"改"四个数"；§2.5 写明过渡段的前引也删了。figures/make_compact.py 顶部注释同步。

**第二轮（核对第一轮改动 + 通读）：**
- 方法名残留：E.1.2 表注 "token-level PPO"、Appendix D 两处 "GPT-2 PPO" 统一为 "token-level GRPO/PPO"。E.1.2 表：SFT Sudoku 行两个并列的 1.7 只加粗最早的；K&K dropout-on 行按表注规则加粗第 0 步（RL 没有一格超过起点）。E.2.5 "as a real signal does" 限定到 Countdown。
- 本 MD：Ablations 与主结果两条写法整理的描述更新为第一轮改写后的状态；改动位置清单注明数字已删。
- §5 主结果："wherever the supervised model leaves headroom" 对 Sudoku 不成立（GPT-2 的 SFT 只有 1.2 却只涨 0.2），"an order of magnitude" 对 Game24（7.9×）也偏大；改为 "roughly ten times as much as CFRL-GRPO on Countdown and Game24, the two tasks where its supervised model is neither saturated nor near zero"；"several times as much"（三任务里两个只有 2.5×）改 "at least twice as much"。Ablations 的 "(not tabulated)" 改 "(not in Table 3; Appendix E.3.10)"（E.3.10 表里有）。frozen 结论补 E.3.7 指针。Table 2 表注 "---: not run" 改 "not measured"。
- §4.2 "the critic's trainable part" 与 "fine-tuned once with the head before RL" 矛盾，改 "the part of the critic refit during RL"。Appendix D：critic 预训练的选 epoch 分数对 scalar critic 没定义，补一句"pooled value 代替 minimum"；E.3.7 四个采样器的数值加标签；E.2.6 "the random step does nothing" 改 "moves accuracy by under a point"（Λ/64 处 +0.7，区间不含零）。E.2.5：Game24 的 excess SNR 从 0.032 降到 0.022，恰好预测 cosine 在 0.34 处的平台，改为这样写，不再说 "below the ≈0.5 prediction"。E.1.2 表头 "PPO iteration" 改 "RL iteration"，Sudoku SFT "at epoch 20" 改 "epochs 18–20 (verifier: 1.2 at epoch 20)"。

