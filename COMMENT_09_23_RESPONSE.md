# 对 comment_09_23.md 的逐条回复（2026-09-23）

说明：逐条对应 `comment_09_23.md` 的编号（这份意见针对的是 experiment_suggestions.md 第一版改完后的文本）。每条给"同意 / 不同意 / 部分同意"；同意的写 main 里怎么改，不同意的给理由。原则同上一份回复：与 $s$ 记号有关的建议一律不采纳（作者决定）；页数不作为约束，不为省行删内容；改完过两轮 subagent 审查，发现的问题一并修，列在文末。

## Section 2

1. **不同意。** 不删 "classifier-free guidance" 四个字。没有这个名词，"(Appendix D)" 的指针读者不知道指什么；省一行不是目标。
2. **无需处理。** "it is formed by the re-noising of step $k$ …" 这句在撤回 $s$ 重构时已经删掉了。
3. **不同意。** 四个常数的定义式保留显示式。页数不是约束，显示式比塞进行内清楚。
4. **部分同意。** 采纳建议的精炼写法，审查后定稿为："Here $\alpha_k$ retains part of the previous state, lowering its signal coefficient from $t_{k-1}$ to $\tilde t_k$, where the denoiser is queried, and $\sigma_k$ is the injected noise scale that ELF's convention sets for this $\alpha_k$ (Appendix B.1). The partial step then moves a fraction $\lambda_k$ from $y_k$ toward the estimate, reaching $t_k$." $\sigma_k$ 不能一字不提（Liping 上一轮明确要求解释四个常数），系数相加的推导移到 Appendix B.1（新加标签 `app:kernels`）。
5. **同意。** 改为 "when $\gamma=0$ the sampler reduces to a deterministic ODE sampler with $y_k=z_{k-1}$"。
6. **部分同意。** $\tau$ 在 eq. (2) 之前未定义是对的：现在写 "the density of the trajectory $\tau=(y_1,\dots,y_{K-1})$ factorises as" 再给 (2)，(2) 之后原来的定义句删掉。把 $\tau$ 和 (2) 定义在 $s$ 上：不同意，属于作者已否决的 $s$ 系列；(2) 是 $y_k$ 给定 $s_{k-1}$ 的高斯密度，写成 $s_{k,i}$ 反而要解释第二个分量是确定性的。
7. **不同意。** (a) $J$ 里的 $\mathbb E_c$ 和 $r(w,c)$ 里的 $c$ 保留：verifier 依赖 prompt（目标数、谜面），value (4) 条件在 $c$ 上；Appendix B 的 Proposition 3 和"按 prompt 的 bound 与合并轮次"论证都要区分 $J_c$ 和 $J$，去掉 $c$ 会让附录无法陈述。(b) 建议的 value 段照 main2 写在 $(s_k,t_k)$ 上并带 $t_k$：$t_k$ 由 $k$ 决定，是冗余条件；Bernoulli 说明上一版已经加进 (4)。

## Section 3

1. **同意。** 缩为 "the Gaussian transitions $y_k\mid s_{k-1}\sim\pi^k_\theta$, the factors of (2)"，不再重写高斯。
2. **部分同意。** 删掉 "the policy is not anchored to the initial model" 和带附录细节的半句。保留一句动机："a programmatic verifier leaves little to exploit, and in our runs the anchor only held the policy back (Appendix E.2.2)"。理由：作者要求方法节写动机；只写 "we follow them" 等于没有理由。
3. **不同意（暂不引用）。** 仍无法核实这篇 workshop 论文的内容：openreview 页面和 API 都跳到人机验证页，Semantic Scholar 限流，arXiv 搜不到同名标题。未核实的文献不引；coauthor 若能确认原文有"单步 on-policy 时裁剪不激活"这一结论，加一句 "see also" 即可。
4. **部分同意。** 保留 "At $\theta=\theta_{\rm old}$ every ratio equals one, so the clip is inactive … the first step on (5) is exactly a descent step on (6), and ratio and clipping matter only from the second step on (Proposition 1)"，这是作者要求的"从原始 GRPO 推到我们版本"的关键一句。batch 复用的解释缩成一句指针："What re-using a batch with ratios and clipping buys is task-dependent (Appendix E.2.2)"。
5. **不同意。** eq. (6)(7) 里的 $y$ 不换成 $s$；$s$ 系列。(7) 的 score 正是对 $y_{k,i}$ 的高斯密度求导得到的，写成 $s$ 就看不出噪声因子从哪来。

## Section 4

0. **同意。** Figure 1 图注缩短："Where the reward lands: (a) on the token axis in an AR model; (b) on the step axis in RL for diffusion and flow models; (c) on both in CFRL-SVG-2D: $R_{k,i}$ for flow step $k$ and position $i$, each an estimate of the whole response's outcome, pooled into $V_\phi$."
1. **同意。** §4 开头改为 "both variants learn a critic $V_\phi$ and ascend it through one transition of the chain"，log-odds 留给 §4.1 的 "The estimator" 段。
2. **部分同意。** 分解段首句改为 "The true value $V^\star_k(s_k)$ summarises the expected reward of all continuations from $s_k$, and by the recursion (4) $J=\mathbb E_c\,\mathbb E[V^\star_k(s_k)]$ at every $k$. One parameter copy per transition therefore splits $\nabla_\theta J$ into one term per transition, each the one-step value gradient of SVG (Heess et al., 2015) with the true value; with a critic in place of $V^\star_k$, one such term becomes an estimator." 没有写 "unrolling the SVG recursion"：我们的分解是参数副本加 tower property 的全导数拆分，每一项才是 SVG 的单步 value gradient，不是 SVG 论文那个递推本身。
3. **不同意。** eq. (8) 不写成 $\nabla_{s_k}V^\star_k\,\partial s_k/\partial\theta_k$，不去掉 $y_k$、$\hat x_{k-1}$ 两个通道的分解。两个通道正是与 REINFORCE 的区别所在：第二通道（经估计）在 (6) 里没有对应项，并且占大部分 pathwise 梯度（Appendix E.3.5）；写成一个 $s$ 的梯度就把这一点藏起来了。(8) 下面的段落也不缩：它解释 score 因子 $V\varepsilon/\sigma$ 与导数 $\partial V/\partial y$ 的方差差别，是 §3 到 §4 的论证衔接。
4. **不同意。** §4.2 首句保留 "$s_k=(y_k,\hat x_{k-1})$" 和 "the estimate $\hat x_{k-1}$ is the informative part…" 两句。critic 的设计理由就是"估计分量携带结果信息、denoiser 是读它的网络"，Table 2 的消融（去掉 $\hat x_{k-1}$ 输入）直接对应这句；建议的替换句说的是同一件事但去掉了分量名，之后的消融就没有指代对象。
5. **不同意。** 特征不写成 $\hat X_{\bar\theta}(s_k,\tilde t_k)_i$：与 eq. (1) 的调用签名 $\hat X_\theta(z,t;\hat x_{\rm prev})$ 不一致，且 $s$ 系列。
6. **不同意。** "$s_{k,i}=(y_{k,i},\hat x_{k-1,i})$" 这个定义保留，它是上一轮审查要求补的（$s_{k,i}$ 此前未定义）；只写 $s_{k,i}$ 又回到未定义。

## Section 5

1. **同意。** Table 1 的 GPT-2 和 LLaDA 的 RL 行也加相对 base 的增益（+9.7/+0.2/+0.3/+5.3；+1.8/+0.0/+1.5/+3.4），表注改为 "RL rows carry the gain over their model's task-SFT row in parentheses"（base 在文中专指 ELF-B 的 SFT 模型）。
2. **同意。** Table 1 右面板底部加图例（四条：score-function、pathwise (2D)、control / random、GPT-2 (Countdown)，6.4pt），图注删掉线型键，只留 "bands: paired 95% intervals"；面板高度从 1.96 英寸加到 2.3 英寸。图注同时改正两处：右面板不全是 Game24（GPT-2 那条是 Countdown），"gain over the base" 改为 "gain over their model's task-SFT row"（base 在文中指 ELF-B 的 SFT 模型）。
3. **同意（小改）。** Setup 按建议措辞重写。保留两处建议里删掉的内容：related work 的附录指针（相关工作在附录，正文必须有一处指向）；$\gamma56$@64 记号的定义和两条便宜链的名字（上一轮审查要求，正文首次出现处必须说明 churn 和步数）。
4. **同意（小改）。** Main result 按建议重写；保留 "which differs from it only in the critic" 半句，这是 scalar 与 2D 的关键对照；"the same estimator" 写成 "the same score-function estimator"。
5. **部分同意。** 采用建议的结构和大部分措辞。两处不采纳：(a) "the score-function gain is small and indistinguishable from zero" 与表格矛盾（Game24 在 $\Lambda/32$、$\Lambda/16$ 处区间不含零），保留上一轮审查后的准确表述："Up to a sixteenth of $\Lambda$ both steps change accuracy by about a point at most; from a quarter to a half of $\Lambda$ the pathwise step raises it with an interval that excludes zero … while the score-function step gains under two points on Game24 and nothing on Countdown, with intervals that include zero"；(b) "training with the deployment chain improves CFRL-GRPO by a few points" 与 E.1.3 不符（换链几乎不变），写成 "move CFRL-GRPO by a few points on some tasks"。
6. **部分同意。** 采用建议的措辞。两处不采纳："by at least a third"（0.93→0.65 是 30%）保留 "by about a third or more"；"as predicted by (9)" 保留 "as the bias (9) leads one to expect"（bias 论证是启发式，不是预测）。
7. **同意。** Table 2 右面板底部加图例（batch success、held-out accuracy、critic value 右轴、base，6.4pt），图注删掉全部线型键，面板高度 1.2 英寸加到 1.85 英寸。
8. **部分同意。** 采用建议的结构（先图、再固定策略下 critic 等价、再交换实验、再结论）。"other tasks show similar pattern" 与 E.3.3 不符（Countdown 的 scalar 回落而非变平，Sudoku 的 scalar 在 250 步后回落、2D 在 500 步后回落），保留具体写法。第二段保留原措辞（意思与建议相同）。
9. **同意（几处保留）。** 段名改为 "Ablations"，按建议措辞重写。保留三处："refitting at the window's final step is important" 只在 Countdown 成立，Game24 上五步 buffer 与单步持平（E.3.10），加限定；"mean beats minimum" 后保留指向 Appendix C.3 的半句（为什么搜索用最小值）；"as the bias (9) leads one to expect" 不写成 "consistent with"。最后一句改为"在便宜链下训练的 run 在便宜链上有增益、带到部署链的很少"，因为 Table 3 只给部署链读数，原句读者看不出增益是在哪条链上取得的。
10. **同意。** loop 段按建议措辞重写，保留 "on K&K the first round re-uses the training problems without their labels" 的半句（Appendix A 与 D 的一致性要求）。

## Section 6

1. **部分同意。** 缩短但不全删：保留 "peaks above the base at every seed"（论文的可靠性主张）和 "the loop carries its verified samples into a cheap chain on two of the three tasks with a pool"（第三个贡献），去掉 "on three tasks"。
2. **同意。** 首句加 "and an amortisation loop"。
3. **同意。** limitation 写成 "CFRL-GRPO trained under a cheap 16-step chain rather than the deployment chain"。

## 审查后额外修改的地方

第一轮（两个 agent：正文逻辑与数字核对；MD 与 tex 一致性）：
- §5 主结果："同一估计量对 AR 更有效"原来只基于 Countdown，且指针指错；改为 "GPT-2 gains ten points on Countdown and five on Game24, against one and two and a half for CFRL-GRPO: where both have headroom …"，指向 Appendix E.1.2、E.2.5（Sudoku、K&K 上 GPT-2 已无提升空间）。"gains several times more" 改为 "two to seven times as much"；Sudoku 的 scalar 写成"每个 seed 在 250 步达峰后回落"（seed 2 的峰值高于 base，不能说"每个 seed 都下降"）。
- §5 2D 段：Sudoku 两条曲线的回落时刻分别写（scalar 250 步、2D 500 步，E.3.3）；"while the policies are still close" 改为 "within a few points"（250 步时差约三分）。
- §5 loop 段："Sudoku covers only puzzles the converged model already solves" 没有附录支撑，改为 E.4.1 实际给出的事实：一轮覆盖约四分之三的 pool 且不变，用 pool 的 gold 解在全覆盖下训练也只变不到一分；K&K 加上"从 SFT base 直接跑 loop 也到 99.6"（RL 不是那里接近满分的原因）。
- §5 上限段："not the limitation" 改为 "unlikely to be the limitation"（证据是几个变体、部分单 seed）；补一句 Λ 处 Game24 的 pathwise 增益回落（不重拟合的 critic 不能跟那么远）。Ablations 段最后一句见第 9 条。
- §4.1 首句改精确：由 (4) 得 $J=\mathbb E_c\mathbb E[V^\star_k(s_k)]$ 对每个 $k$ 成立，参数副本把 $\nabla_\theta J$ 拆成逐转移项，每项是 SVG 的单步 value gradient（真值），换成 critic 即得估计量。
- §2：$\sigma_k$ 那半句消歧（"lowering its signal coefficient from $t_{k-1}$ to $\tilde t_k$ … the injected noise scale that ELF's convention sets for this $\alpha_k$"）；pathwise 恒等式里的测试函数 $g$ 改为 $\psi$（$g$ 是 §3 的 rollout 下标）；score-function 恒等式注明"读出按 §3 采样"。
- §3：同一段里三处指向 Appendix E.2.2，删掉第一处；"the anchor only held the policy back" 加 "in our runs"（E.2.2 是两个任务各一个 seed）。
- §6："keeps the policy improving" 加 "longer"（Sudoku 的 2D seed 在 500 步后回落）。
- 版面：Algorithm 1 改回 `[t]`（`[H]` 放不进第 6 页时把第 6 页撑出空隙）；Table 1 图注压回 6 行；两个小面板的图例字号提到 6.4pt，Table 2 面板的图例加上 base 线。

第二轮（两个 agent：核对第一轮修改并通读；MD 审计）：
- §5 主结果：GPT-2 与 flow 的比较不能再用 Game24——E.2.2 里带比值和裁剪的两步版本（即 GPT-2 用的 PPO recipe）在 Game24 上是 +5.0，与 GPT-2 的 +5.3 相当；改为"Countdown 上 GPT-2 的 PPO 得十分、CFRL-GRPO 得一分、两步版本不到两分；Game24 上两步版本与 GPT-2 持平，所以'同一估计量对 AR 更有效'的证据是 Countdown 和 Table 1 右面板的梯度测量"。"gains only a few points on each task" 改为 "at most four points on any task"（Countdown 只有 +0.9）；"two to seven times" 改为 "two and a half to seven times"；LLaDA 写成 "adds at most three and a half points"（Game24 上 +3.4，比 CFRL-GRPO 多）。
- §5 2D 段和主结果：Sudoku 的 scalar critic 从未明显高于 base（各 seed 在第一个检查点 250 步达峰、峰值不超过 base 一分左右），不再写成"250 步后回落"。
- §5 loop 段：Sudoku 不是 coverage 受限的例子（gold 解全覆盖也只变不到一分），改写为"该 pool 已无可教"；Countdown 的 critic 引导轮次注明链和引导同时改变。
- §5 上限段：Λ 处的回落只发生在 Game24（Countdown 在 Λ 处仍上升），写明并把"原因"改为"与…相容"。
- §4.1：逐转移项"每一项是 SVG 的单步 value gradient"限定为最后两个转移之前的项（B.4 说明最后两处交换不成立）。
- §2："with the readout sampled as there" 改为 "with the decoder readout sampled (§3)"。
- 版面：两个小面板的图例源字号提到 7.0 / 7.2pt（缩放后 ≥ 6.4pt），Table 2 面板图例改单列。
