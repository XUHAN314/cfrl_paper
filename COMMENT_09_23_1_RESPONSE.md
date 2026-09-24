# 对 comment_09_23_1.md 的逐条回复（2026-09-23 晚）

说明：逐条对应 `comment_09_23_1.md` 的编号（这批意见针对的是 comment_09_23 那轮改完后的文本）。每条给"同意 / 不同意 / 部分同意"；同意的写 main 里怎么改，不同意的给理由。原则同前：与 $s$ 记号有关的建议不采纳（作者决定）；页数不作为约束；改完过两轮 subagent 审查，发现的问题一并修，列在文末。

## Section 2

1. **同意。** 改为 "A program verifier assigns a binary reward $r=r(w,c)\in\{0,1\}$ by checking every component of the decoded response."，删掉举例和"是随机变量"那半句（后者由 eq. (4) 的 Bernoulli 形式表达）。
2. **部分同意。** 删掉 "the critic $V_\phi$ of §4 will approximate its log-odds"，$V_\phi$ 留到 §4 再说。保留括号里的 "the star marks the true value" 四个词：星号是 Liping 上一轮要求加的，不说明读者不知道它标什么。
3. **部分同意。** 按建议重写，递推写成显式的 $V^\star_{k-1}(s)=\mathbb E_{\varepsilon_k}\big[V^\star_k\big(\alpha_kD_\theta(s)+\sigma_k\varepsilon_k,\ \hat X_\theta(s)\big)\big]$，范围写 $2\le k\le K-1$（$k=1$ 没有 denoiser 调用），去掉笔误 "with while"。但建议里的 "the re-noising Gaussian-smooths its dependence on $y_k$" 主语不对：这一步的 $\varepsilon_k$ 平滑的是 $V^\star_{k-1}$ 对均值 $D_\theta(s)$ 的依赖，$V^\star_k$ 对 $y_k$ 的平滑来自下一次 re-noising（§4.1 讲）。现在写成 "The last value, $V^\star_{K-1}(s)=r(w_\theta(s),c)$, is piecewise constant in $s$, and $J=\mathbb E_c\,\mathbb E[V^\star_1(s_1)]$. For $2\le k\le K-1$ the tower property gives … so $V^\star_k$ depends on $\theta$ through the later transitions. The re-noising averages $V^\star_k$ over a Gaussian in $y_k$, so $V^\star_{k-1}$ depends smoothly on the mean $D_\theta(s)$, while the self-conditioning estimate $\hat X_\theta(s)$ passes through deterministically (§4.1, Appendix B.4)"。

## Section 3

1. **同意。** 首句改为 "The exact Gaussian transition densities in (2) enable a direct adaptation of GRPO (Shao et al., 2024) to continuous flow generation."
2. **部分同意。** "in our runs the anchor only held the policy back (Appendix E.2.2)" 从 §3 删掉；保留动机半句 "a programmatic verifier leaves little for an unanchored policy to exploit"（方法节要有理由，只写 "we follow them" 不够）。经验结论移到 §5 的"为什么受限"段：变体列表里加 "restoring the anchor removes even those points (Appendix E.2.2)"。
3. **部分同意。** 不再说"第二步"，改为 "ratio and clipping only matter when a batch is re-used, which CFRL-GRPO does not do (Proposition 1; off-policy variants: Appendix E.2.2)"。这句不能整个删：它是 Proposition 1 的结论，也是作者要求的"从原始 GRPO 推到我们版本"的收尾。
4. **同意。** "What re-using a batch … is task-dependent" 从 §3 删掉，并入第 3 条的附录指针。
5. （空）

## Section 4

1. **部分同意（记号改，"截断近似"不改），这条改动最大。** 实现里只有一套 $\theta$，"每个转移一份参数副本"是记账手法，但读起来像架构，Liping 也不喜欢 $\theta_k$。现在不再谈副本：定义 $\nabla^{(k)}_\theta$ 为"只经第 $k$ 个转移那次 denoiser 调用、其余调用视为常数、对 $\theta$ 求的梯度"，正文写明 "this is bookkeeping, not a copy of the parameters"；链式法则给出精确恒等式 $\nabla_\theta J=\sum_{k=2}^{K}\nabla^{(k)}_\theta J$，$\nabla^{(K)}_\theta$ 收 $s_{K-1}$ 处的调用、最后一步和解码。eq. (8) 写成 $\nabla^{(k)}_\theta J=\mathbb E[\nabla^{(k)}_\theta V^\star_k(s_k)]$，Jacobian 写 $\nabla^{(k)}_\theta\hat x_{k-1}$；eq. (9) 的 $\hat g_{\rm PW}$、$b_k$，Algorithm 1 前后的文字，Appendix B.4 Step 1 到 Step 5 的 (16)(17)(18)(20)(21)，Appendix C.1 全部同步替换（$D_{\theta_k}$、$\hat X_{\theta_k}$ 改回 $D_\theta$、$\hat X_\theta$，"the copy $\theta_K$" 改为 "the term $k=K$"）。关于"eq. (8) 第一部分应写成截断近似"：单个 $\nabla^{(k)}_\theta J$ 是精确的，不是近似；近似发生在两处并已写明——估计量只取窗口内随机一个 $k$ 的这一项（"Only the call at $s_{k-1}$ is differentiated"），以及用 critic 代替 $V^\star_k$（bias $b_k$）。

## Section 5

1. **同意。** 主结果只讲 Table 1 里有的东西（第一句改为 "gains at most about four points on any task"，因为 K&K 是 +4.1）："GPT-2's PPO gains ten points on Countdown against one for CFRL-GRPO, and LLaDA's masked-diffusion estimator adds at most three and a half points (further comparisons: Appendices E.1.2, E.2.2, E.1.4)"，不再提两步 recipe，也不再下"同一估计量对 AR 更有效"的一般结论（它只有 Countdown 支撑）。
2. **部分同意（图例已改；单数据集等实验核对）。** 图例改成两行两列，"score-function" 改 "CFRL-GRPO"，"control / random" 改 "control"。GPT-2 那条标 Countdown 是因为 GPT-2 的梯度测量当时只在 Countdown 做过（Appendix E.2.5），flow 的在 Game24。同意面板只用一个数据集：已在 Game24 上重跑 GPT-2 的 half-split cosine（作业 4246904，已完成；脚本、prompt 数和 seed 与 Countdown 的测量一致，N=256/1024/2048）。结果待核对：GPT-2 token-GRPO 梯度的 half-split cosine 为 0.09 / 0.35 / 0.34（Countdown 上是 0.05 / 0.17 / 0.27），方向一致，都随 N 增大而离开控制线。按"新实验结果核对后再放"的规则，面板暂时保留 Countdown 那条并在图例里标出，正文写 "(measured on Countdown)"；核对通过后换成全 Game24 并删掉这个标注。
3. **部分同意。** "blue: this paper's recipe" 确实不对，三个变体都是我们的。但表里留蓝色而图注不解释，审稿人会挑；改为三个词 "blue: recommended variant"，Table 4 和附录里带蓝色行的两张表（E.3.10 的 ablation 轨迹表、E.4.2 的 cheap-sampler 表）同步。
4. **同意。** 删掉 "Up to a sixteenth of Λ both steps change accuracy by about a point at most"。
5. **同意。** 单步干预只讲 Game24：Countdown 只留指针 "(Countdown: Appendix E.2.6; discrete baselines: Appendix E.2.5)"，"on Countdown as well"、"and nothing on Countdown"、"(on Countdown it does not)" 都删；Λ 处 Game24 增益回落那半句留一句。
6. **同意。** 其他任务压成一句："on Countdown and K&K the 2D critic likewise ends ahead of the scalar one, whose policy turns back or flattens; on Sudoku both turn back, the scalar critic from its first checkpoint and the 2D critic from a higher peak (Appendix E.3.3)"。"ends ahead" 指最后一个 checkpoint 的三 seed 均值：Countdown 50.0 vs 45.9、K&K 79.6 vs 77.1（2D vs scalar；Sudoku 也是 65.1 vs 62.7，但两者都低于 base，所以单独说）；峰值同样是 2D 高：50.2/48.2、71.0/66.3、79.6/77.6。
7. **同意。** 改为 "Neither critic remains effective when frozen, so the advantage arises from how these choices interact with the refit (Appendix E.3.4)"。
8. **同意。** 改为 "On Game24, refitting the head-before-mean critic with the pooled loss alone is sufficient; on Countdown, where the scalar critic loses discrimination as its policy turns, the per-position loss is also necessary (Appendix E.3.10)"。
9. **同意。** Table 3 只留 Game24：五步 buffer 的 Countdown 行删掉（它在 Appendix E.3.10 的 block B 里），图注去掉 "Countdown where marked"，行号说明改为 "rows 1–2, 3–4, 5"；Appendix E.3.10 的图注同步改为 "rows 1–2, 3–4 and 5"。正文的 Ablations 段只讨论 Game24 的数字；Countdown 只剩两处指针性的话，有意保留："(the rest, and Countdown: Appendix E.3.10)"，以及 "refitting at the window's last step is neutral here but needed on Countdown, where the outcome is not readable before the middle of the chain (§4.3; Appendix E.3.10)"，因为 §4.3 选最后一步 refit 的理由就来自 Countdown，不说读者会问为什么 recipe 里有一个在 Game24 上中性的选择。步长那句不再提 Sudoku。
10. **同意。** K&K 第一轮的括号说明删掉，只留 "(pools and Game24: Appendices D, E.4.4)"。
11. **同意。** 改为 "On Countdown and K&K, the loop raises the cheap chain to or beyond the base model's deployment accuracy, transferring the gains across chains."
12. **部分同意（留三个词）。** 改为 "The loop is limited by both coverage and pool novelty (Proposition 3; Appendix C.3): K&K reaches near-perfect accuracy with broad coverage, from any start, whereas on Sudoku even full-coverage gold solutions add only about a point." "from any start" 保留，因为从 SFT base 直接跑 loop 也到 99.6（E.4.1），不写读者会把 K&K 的满分归功于 RL。
13. **同意。** 改为 "On Countdown, plain rounds reduce coverage and partially erase the RL gain; rounds drawn with the critic-guided cheap chain restore both (chain and guidance change together; Appendix E.4.1)."

## 审查后额外修改的地方

改完后两轮 subagent 审查（每轮两位：一位审 §2–§5 与附录 B/C 的数学、记号、连贯性并核对 §5 每个结论对应的表；一位审本 MD 与 tex 是否一致）。

**第一轮（20 条 + 9 条）：**
- 记号残留：Appendix B.4 Step 1 的 "the terms by which $\theta_k$ moves the means…" 和 "The $\theta_K$ channel…"、C.1 的 $\sum_k\partial\tilde J_k/\partial\theta_k$ 改成 $\nabla^{(k)}_\theta$ 形式；B.1 的 "$\partial_\theta V_\phi(s_k)$ is computed by the chain rule" 改为 $\nabla^{(k)}_\theta V_\phi(s_k)$（只对 $s_{k-1}$ 处那次调用求导，与 eq. (9) 一致）。
- §2：$\hat X_\theta(s_k)$ 这个单参数简写原来只在附录 B.4 定义，现在在 estimate 行写 "$=:\hat X_\theta(s_k)$"。值函数段 "the re-noising Gaussian-smooths its dependence on $y_k$" 主语不准：$\varepsilon_k$ 平滑的是 $V^\star_{k-1}$ 对均值 $D_\theta(s)$ 的依赖，不是 $V^\star_k$ 对 $y_k$ 的依赖（后者靠下一次 re-noising，§4.1 讲）。改为 "The re-noising averages $V^\star_k$ over a Gaussian in $y_k$, so $V^\star_{k-1}$ depends smoothly on the mean $D_\theta(s)$, while the self-conditioning estimate $\hat X_\theta(s)$ passes through deterministically (§4.1, Appendix B.4)"。"$V^\star_{K-1}(s)=\dots$ is piecewise constant, while $J=\dots$" 改为 "The last value, …, is piecewise constant in $s$, and $J=\dots$"。
- §3：“$\ell_i$ the decoder logits at $z_K$, which replaces …" 的 which 指代不清，改为 "; sampling them replaces …"。
- §4.1："by the recursion (4)" 改为 "by the tower property applied to (4)"（(4) 是定义，递推本身没有编号）；"For every transition before the last two, the term is …" 改为 "For $2\le k\le K-2$ the term is …, as we now show"；坐标比较那句加 "up to the common factor $\alpha_k\lambda_{k-1}$"（eq. (7)/(17) 与 eq. (8) 里两个因子都带这个公共因子，原句略去了），Appendix B.4 Step 3 同句同改。§4.4 "(tested on Countdown, …)" 去掉数据集名（方法节规则）。
- Appendix B.4：Step 1 之前 "…and the decode, the term $k=K$ of Step 1, and through no transition" 悬垂，改为括号 "(the term $k=K$ of Step 1 below)"；eq. (21) 不再重复定义 $b_k$，标 "as in (9)"。
- §5：CFRL-GRPO "gains at most four points" 与 K&K 的 +4.1 不符，改 "at most about four points"（本 MD §5.1 已同步）。2D-critic 段 "on the other tasks the 2D critic likewise ends ahead" 会被读成高于 base，而 Sudoku 两个 critic 都在 1000 步前回落（scalar 从 250 步、2D 从 500 步的更高峰值回落），改为 "on Countdown and K&K the 2D critic likewise ends ahead of the scalar one, whose policy turns back or flattens; on Sudoku both turn back, the scalar critic from its first checkpoint and the 2D critic from a higher peak"（本 MD §5.6 同步）。Ablations 段步长句加 "Here"（Sudoku 上 decaying 步长的 2D critic 也在 500 步后回落，E.3.3；E.3.9 原来那句 "holds on every task and seed" 与此矛盾，已改）。Loop 段 "Sudoku gains nothing" 改 "gains at most about a point"（gold 对照 +1.3 cheap / +0.9 deploy）。SNR 面板的 Countdown 指针改精确：flow 模型的 Countdown 测量只在 E.2.6，E.2.5 是 GPT-2/LLaDA 的，写成 "(Countdown: Appendix E.2.6; discrete baselines: Appendix E.2.5)"。Table 3 图注写明 "recipe blocks A, B and C of Appendix D"。
- 附录 D：E.3.10 表图注 "rows 1–3, 4–5 and 6" 改 "rows 1–2, 3–4 and 5"；两处 "blue: this paper's recipe"（E.3.10 表、E.4.2 表）改 "blue: recommended variant"。
- 本 MD：§4.1、§5.12 的立场从"同意"改为"部分同意"（各保留一点没照改）；§5.2 更新作业 4246904 已完成及待核对数值；§5.9 写明有意保留的两处 Countdown 指针。

**第二轮（核对第一轮改动 + 通读 §2–§5、附录 B/C）：**
- 附录 E.3.9（学习率）"At $10^{-6}$ the 2D critic holds on every task and seed" 与 E.3.3/§5 的 Sudoku 描述矛盾（Sudoku 三个 2D seed 都在 500 步后回落、1000 步低于 base），改为 "holds its gain on Countdown, K&K and Game24; on Sudoku it peaks at 500 iterations and turns back"。
- 附录 E.4.1 "On Sudoku one round … changes nothing at any sampler, as the gold-solution control confirms" 与正文 "at most about a point" 不一致（gold 对照 +1.3 cheap / +0.9 deploy），改为 "changes accuracy by at most about a point at any sampler, as the gold-solution control confirms"。
- 本 MD：§2.3、§5.2 立场改为部分同意；§5.1、§5.5、§5.6、§5.12 的引文同步到 tex 现在的措辞；Countdown 2D 终点均值按表格写 50.0；蓝色行表格是 E.4.2 不是 E.1.2；学习率附录编号 E.3.9。
- §2："adds fresh noise to the state before every denoiser call" 漏了最后一步（有调用、无加噪），改 "before every denoiser call but the last"；$s_k$ 的定义移到它第一次出现的 estimate 行；"depends on $\theta$ through the later transitions" 对 $k=K-1$ 不成立（没有后续转移，只有后续调用），改 "through the later denoiser calls"。
- §4.1：分解求和到 $k=K$，但正文只讲 $2\le k\le K-2$，补一句 "the terms $k=K-1,K$ pass through the piecewise-constant $r\circ w_\theta$ and give a pathwise estimator no signal (Appendix B.4)"。
- Appendix B.1："the score-function estimate of $\partial_\theta\mathbb E[r]$ is …" 不是完整的 score（缺 self-conditioning 项和 readout 因子，Step 1 自己也这么说），改 "the transition part of the implemented score-function gradient is …"。
- Appendix E.3.3（轨迹）："On K&K the Game24 shape holds" 不对：K&K 跑 1000 步，2D critic 一直升到最后（74.5/77.8/78.8/79.6），scalar 500 步后平在 77 左右；改写。E.3.9 "the 4k buffer of that phase" 会让人以为 Table 3 第 3 行（3e-6）的 buffer 与 block B 不同；查日志该 run 的 buffer 饱和在 16,320，即已是 16k，句子改为 "the 4k buffer of the first joint runs (the Game24 run above already used 16k)"。E.4.1 Sudoku 一轮 loop 的描述与 gold 对照数值一致化（+1.3 / +0.9），正文 §5 改为 "even full-coverage gold solutions add only about a point"。E.4.2 表注 "that policy after the loop" 改 "its best seed after the loop"（loop 从最好 seed 起跑，不是三 seed 均值）。Appendix D 算力：Sudoku 32 步 ×1000 迭代约 2M 次调用，GRPO 只便宜 1.7 倍，句中补 "($2$M on Sudoku)"、"(under two on Sudoku)"。
- **不采纳**：审稿 agent 建议把 token 因子 $\pi^i_\theta$ 改成 $\pi^{w_i}_\theta$ 以免与 $\pi^k_\theta$ 混淆。$k$ 与 $i$ 是全文固定的转移/位置下标，§3 已写明 "$u$ a transition $k$ or a token $i$"，eq. (6)、Proposition 1、附录 E.2.2 都用这套记号，改动收益小、牵连大。

