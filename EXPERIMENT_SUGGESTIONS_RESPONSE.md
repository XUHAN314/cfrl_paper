# 对 experiment_suggestions.md 的逐条回复（2026-09-23）

说明：逐条对应 `experiment_suggestions.md` 的编号。"同意"表示已按建议改入 `main.tex` 及其输入文件；"部分接受"表示按建议的意图改了但方式不同；"不同意"给出理由。改动全部在 `sections/*.tex`（不含 `*1.tex`、`*2.tex`）和 `appendix/*.tex`。正文总长放宽到 10 页：方法部分到第 7 页上部结束（Algorithm 1 和 §4.4 在第 7 页），§5 从第 7 页下部开始，Table 1 在第 8 页顶，Conclusion 在第 10 页，参考文献从第 10 页下部开始。改完后过了两轮 subagent 审查（逻辑、全文记号、行文），审查里指出的问题也一并修了，列在文末。

## Section 2

1. **同意。** "time grid" 改为 "time steps $0=t_0<\dots<t_K=1$"。
2. **同意。** "steps $\Delta_k$" 改为 "step lengths $\Delta_k=t_k-t_{k-1}$"。
3. **部分接受。** CFG 的公式和权重移到 Appendix D（ELF-B 的模型描述段，注明 $\omega=2$、critic 的拷贝不用 guidance）。正文只留半句 "A call may combine a prompt-conditioned and an unconditioned evaluation (classifier-free guidance, Appendix D); we treat the combination as one denoiser call"。留半句的原因：§4.2 说 critic 的拷贝 "evaluated without guidance"，Appendix C 的显存论证依赖"一次调用是两次评估"，读者需要知道这个词指什么。
4. **同意。** 改为 "between scaled Gaussian noise $\varepsilon\sim\mathcal N(0,I)$ and the clean embeddings $x$, for $t\in[0,1]$ and a fixed noise scale $c_\sigma$"。
5. **同意。** 采样器段重写：先说采样器在时间步上运行并在每次调用前加噪（定义 re-noising），再定义状态 $s_k=(y_k,\hat x_{k-1})$（由第 $k$ 步的再加噪形成，是该步 denoiser 调用的完整输入），然后才写起点 $y_0=c_\sigma\varepsilon_0$。
6. **部分接受。** 保留 "re-noise"（扩散采样文献里的常用说法），但在首次出现处用斜体定义："adds fresh noise to the state before every denoiser call (\emph{re-noising})"。
7. **同意。** 三行更新式之后紧跟四个常数的定义，再逐个解释作用：$\alpha_k$ 是保留的上一状态比例，它把信号系数从 $t_{k-1}$ 降到 $\alpha_kt_{k-1}=\tilde t_k$，denoiser 因此在 $\tilde t_k$ 被查询；$\sigma_k$ 是注入噪声的尺度，使噪声系数加起来等于插值在 $\tilde t_k$ 的噪声系数（$\alpha_k(1-t_{k-1})+(1-\alpha_k)=1-\tilde t_k$，ELF 的约定）；$\lambda_k$ 是部分步从 $y_k$ 走向估计的比例，走完后状态到达 $t_k$。
8. **部分接受。** 我读了 main2 的 §2。它把三行更新式压成一行、四个常数塞进一句，比现在更难读，所以没有照搬；采纳了它"先给状态、再给链"的顺序（见第 5、12 条），解释按第 7 条的方式给。
9. **同意。** 读了 main2 的写法。eq. (4) 现在写成 $V^\star_k(s)=\mathbb P(r=1\mid s_k=s,c)$，并注明 $r\mid s_k\sim\mathrm{Bernoulli}(V^\star_k(s_k))$，把"给定中间状态时 reward 是随机变量"写明。没有照抄 main2 把 value 定义在 $y_k$ 上（见第 12 条）。
10. **不同意，改为压缩。** 这一段是全文的概念支点：两个梯度恒等式的对比、"AR 的每一项是类别决策、flow 的每一项是噪声坐标"、以及 Appendix B.4 用到的正则性条件都在这里，§3、§4 和附录都指回它（标签 `sec:estimators`）。删掉后这些内容要在 §4.1 重写，篇幅不省而连贯性受损。现已压缩，只保留两个恒等式、AR 对比和"reward 没有可微路径、中间状态的 value 有"。
11. **同意。** 真值统一写为 $V^\star_k$（大写星号，与 $V_\phi$ 对应），eq. (4) 后注明 "the star marks the true value; the critic $V_\phi$ of §4 will approximate its log-odds $\mathrm{logit}\,V^\star_k$"。全文替换：§2、§4、Appendix B（约 50 处）、Appendix C（17 处），包括 $V^\star_{K-1}$、$V^\star_{k\pm1}$、$V^\star_1$、$\nabla_yV^\star_k$、$\mathrm{logit}\,V^\star_k$。
12. **部分接受。** $s_k$ 现在在采样器段的第二句就定义为状态，链、密度、value 都写在 $s_k$ 上。但 $y$ 和 $\hat x$ 不能全篇消失：(a) §3 的高斯因子是 $y_k$ 在给定 $s_{k-1}$ 下的密度，eq. (6)、(7) 必须写 $y_k$；(b) eq. (8) 的两个通道（经再加噪状态 $y_k$、经估计 $\hat x_{k-1}$）是方法的核心机制，"第二通道占大部分梯度"和 score-function 作用在 $y_k$ 上都要写分量；(c) critic 的输入和 Table 2 的消融（去掉 $\hat x_{k-1}$）也要说明读的是哪部分。所以 §3 的高斯因子、§4.1、§4.2 和 Table 2 保留分量，其他地方只写 $s$。

## Section 3

0. **同意（措辞），部分接受（引用）。** 删掉 "the anchor protects nothing" 那句。现在写 "$\beta=0$: as in group-relative REINFORCE (Yao et al., 2026) and DAPO (Yu et al., 2025), the policy is not anchored to the initial model. Our programmatic verifiers leave little to exploit, and on the two tasks where we tried it, adding the anchor to the clipped two-step form undid its gain and left it no better than CFRL-GRPO (Appendix E.2.2)." 最后半句是 Appendix E.2.2 表格的高层结论（带 anchor 的两步版本回到 base 上方约 1 分，不比 CFRL-GRPO 好），不带数字；审查指出原先写的"消除全部增益"超出了表格证据，已改。引用核实：Yao et al. 是 ICLR 2026 正式论文，我下载了 PDF，正文明确说他们的 GRPO 实验不用对参考模型的 KL 正则并引 DAPO 说这类项常不必要，可以引；DAPO 本来就在 bib 里，它明确去掉 KL 项。Proposition 1 的证明本身不需要引用。
1. **同意。** 现在写 "We write $u$ for a factor of the rollout density, a transition $k$ or a token $i$, and $\pi^u_\theta$ for that factor"，两类因子显式定义为 $\pi^k_\theta:=\mathcal N(\alpha_kD_\theta(s_{k-1}),\sigma_k^2I_d)$ 和 $\pi^i_\theta:=p_\theta(\cdot\mid z_K)$；参考因子 $\pi^u_{\rm ref}:=\pi^u_{\theta_{\rm ref}}$ 也在此定义，不再出现无上标的 $\pi_\theta$、$\pi_{\rm ref}$。$\pi^{\rm AR}$ 在 §2 只用于 AR 模型，带上标，不冲突。
2. **部分接受。** eq. (6) 改用 $\log\pi^k_\theta(y^g_k\mid s^g_{k-1})$ 和 $\log\pi^i_\theta(w^g_i\mid z^g_K)$，不再写出高斯。eq. (7) 处仍展开高斯，因为 "score = 噪声/σ 乘 Jacobian" 这个全文的关键点要从高斯密度直接看出来。
2.5. **不同意。** 与作者 9 月 21 日定的规则冲突："必须先给原始 GRPO 的形式，不能直接跳到我们的版本，不能上来就一个变体版本然后 claim 这是 GRPO"。eq. (5) 是原始 GRPO 目标，是我们版本的出发点，Proposition 1 证明的就是从 (5) 到 (6) 的等价。(6) 已按第 2 条用一般记号写紧凑了。
3. **同意。** 不再用一个 $\nu_u$ 两个值：正文写 "a random subset $\mathcal K$ of the transitions, with weight $\nu_k=1$, and all tokens, with weight $\nu_i=1/|\mathcal G|$, so that the readout counts as one transition"；eq. (6) 本来就是两个显式求和。eq. (5) 保留 $\sum_u\nu_u$ 的一般形式，因为它是标准 GRPO 的写法，$u$ 和 $\nu$ 已在前文定义。
4. **同意。** 删掉 $\delta$，advantage 写成 $\frac{r^g-\mathrm{mean}(r^{1:G})}{\mathrm{std}(r^{1:G})}$；Appendix B.4 关于 std 归一化的备注同步去掉 $\delta$。
5. **部分接受。** 采用了"两个改动"的结构（写作 "CFRL-GRPO makes two changes to (5)"，主语是方法名而非 we，与全文一致；没有写 "flow-specific"，因为 $\beta=0$ 借自 DAPO，并非 flow 特有）、"one on-policy update per batch"、"still depends on $\theta$" 等措辞。两处没采纳：(a) "providing little benefit" 与 Appendix E.2.2 矛盾（逐位置比值加第二次更新在两个任务上有增益），改为 "the ratio of a whole Gaussian transition sums over $|\mathcal G|\,d$ coordinates and leaves the clipping interval after the slightest change of $\theta$, and with per-position ratios what a second step buys is task-dependent (Appendix E.2.2)"；(b) 保留 "ratio and clipping matter only from the second step on (Proposition 1)"，这是与附录证明的衔接。
6. **同意。** eq. (7) 和比值 $\rho^g_k$ 都改成 `\frac{}{}` 形式；比值现在按因子写（转移因子的比值是 $\pi^k_\theta(y^g_k\mid s^g_{k-1})/\pi^k_{\theta_{\rm old}}(y^g_k\mid s^g_{k-1})$），与 eq. (6) 的条件形式一致。
7. **暂不引用。** openreview 链接被浏览器验证页拦住，arXiv 上搜不到同名论文，我无法核实这篇 workshop 论文的内容。"on-policy 时裁剪不激活"是比值恒为 1 的直接推论，Proposition 1 已证明，不需要外部引用支撑。bib 条目 `zhang2026importance` 保留，等 coauthor 确认原文内容后可加一句 "see also"。

## Section 4

1. **不同意（按作者指示）。** Figure 1 不放大。另注：coauthor 已在提交 91de7fe 里更新了图，面板 (c) 现在是 "CFRL-SVG-2D"，全文不再有 "JPC"。
2. **同意。** 开头改为：先说 CFRL-GRPO 无需 critic 但付出噪声因子的代价，"We now introduce two actor-critic variants based on stochastic value gradients (SVG; Heess et al., 2015) that differentiate the chain itself and so remove that factor"（SVG 在此首次展开），然后是 reward 无可微路径、中间状态的 value 有、两个变体都学 log-odds 的 critic 并经一个转移上升。保留了作者 21 日批准的 SVG(1) 结构句，最后一句说两个变体在 critic 分辨率上不同。
3. **部分接受。** §4.1 开头加了段名 `\paragraph{Transition-wise decomposition.}` 和目的句 "We first split $\nabla_\theta J$ into one term per transition, then turn the term at one transition into an estimator with a critic"。没有照抄建议文本，因为其中 "$\theta_K$ parameterizes the final deterministic step and decoder" 是上一轮刚修掉的不精确说法（$s_{K-1}$ 处的最后一次 denoiser 调用也属于 $\theta_K$），正文现在写 "the copy used by the final call at $s_{K-1}$, the last step and the decode"。
4. **同意。** eq. (8) 里去掉了挤占宽度的范围标注（前一句已说明 $k\le K-2$），公式号不再溢出。
5. **同意。** §4.2 首句改为 "We now design the critic. It reads what the value is a function of, ..."。
6. **同意。** eq. (11) 换序，两个变体都以 $V_\phi(s_k)=$ 开头。
7. **同意。** Require 改为 "policy parameters $\theta$; pre-trained critic $V_\phi$ with trainable head parameters $\phi$ on a fixed copy of the denoiser; verifier $r$; sampler $(\gamma,K)$; window $\mathcal W$; buffer $\mathcal B$ and its size $N_{\mathcal B}$; iterations $T$, prompts per iteration $B$, critic minibatch $B'$ and steps $m_{\rm c}$, AUC check every $n_q$ iterations on the newest $N_q$ states with floor $a_0$ and $m'_{\rm c}$ extra steps; learning rates $\eta_\theta,\eta_\phi$"。正文两处 rollout 标明 "under $(\gamma,K)$"，buffer 行写 "keeping the newest $N_{\mathcal B}$"，批下标写 $s^b_k$（与 §3 的 $s^g_k$ 一致，不再与位置下标 $s_{k,i}$ 冲突）。Require 里的每个符号正文都用到，正文没有 Require 之外的符号。
8. **同意。** 改为 "the 2D critic minimises (13), whereas the scalar critic minimises the first term alone. The first term fits $V_\phi$ to $\mathrm{logit}\,V^\star_k$ ...; the second, present only for the 2D critic, makes every reading a calibrated log-odds on its own"。相应地，§4.2 和 §5 里"两个 critic 只在 head 与均值的顺序上不同"改为"在顺序上不同，并由此在损失上不同"。
9. **部分接受。** 采用 "We train the critic jointly with the policy because ..." 的开头，并展开 AUC。三处没照抄：(a) 保留 bias $b_k$ 这条线（"two things move the condition under which the bias vanishes"），它把 §4.3 与 eq. (9)、Appendix B Step 5 连起来，是数学审查者认可的逻辑链；(b) 建议文本 "Both networks are trained under the sampler that will be deployed" 没有 Countdown critic 预训练用便宜链的例外，而上一轮审查已把这个例外列为必须写明；(c) "The copy stays frozen (§5.x)" 指向实验节，现在指向 Table 2，效果相同。另外原来的 "Amortising the policy into the model" 段从 §4.3 里独立为 §4.4，因为它不属于"训练 critic"。

## Section 5

0. **同意。** 开头改为 "We provide empirical support for each claim of §3–§4 through a corresponding table or figure"。
1. **部分接受，实验未启动。** 表注已写明 ELF-B 的 RL 行是三 seed 均值、其他行单次。GPT-2 PPO 补两个 seed 便宜（小模型、600 步，每任务一两个 GPU 小时），值得做；LLaDA-8B 一次评测就要大半天，不值得。按"不说启动不启动"的规则，我没有提交 GPT-2 的运行，等作者确认后可以补。
2. **同意。** Table 1 的三个 ELF-B RL 行每格加了相对 base 的增益（小字括号，例如 Game24 的 "58.8 (+13.8)"，12 个数均已核对）；表改为 `\footnotesize`、左栏加宽到 0.615，右侧面板缩到 0.37，仍在一行内。
3. **部分接受。** Figure 2 两个面板改为图内 legend（左：scalar critic、2D critic、base；右：四条交换曲线），图注相应删掉线型键。Table 1 和 Table 2 右侧的面板在页面上只有约 2 英寸宽，图内 legend 会盖住数据，这两处保留图注里的线型键。
4. **同意（大幅压缩）。** Table 1 图注从 10 行压到约 6 行：右侧面板只说测什么、怎么读，定义放正文和 Appendix E.2.3、E.2.6。Table 2 图注压缩并明确是 "the 2D critic of Algorithm 1"。Figure 2 图注从 5 行压到 3 到 4 行。Table 3、Table 4 图注各压一两行。$D$ 保留：横轴用 $D$ 的分数才读得出"四分之一到一半的 $D$ 就 +4"，换成绝对步长数字没有意义；图注用一句定义它。
5. **同意。** 拆成 "Main result." 和 "Why the score-function variant is capped." 两段，后者带 `sec:exp_estimator` 标签，以"估计量的各种变体（更细的 advantage、每批第二步、部署链）在个别任务上能多几分，但在任何任务上都到不了 CFRL-SVG-2D 的水平；§3 把上限归因于动作空间"开头作为论点（审查指出原先的 "past a few points" 与 Appendix E.2.2 里两步版本在 K&K 上 +6.6 矛盾，已改）。
6. **同意。** 右侧面板的段落先说设置再说结论：half-split cosine 是把 $N$ 条 rollout 分两半、取两个均值梯度的余弦，真信号随 $N$ 增大趋向 1、纯噪声停在 0 附近；单步干预是从 base 沿各均值梯度走一步 AdamW 归一化的步长再评测，检验方向；"raw gradient" 改写为 "the same pathwise gradient without AdamW's per-coordinate normalisation"。
7. **同意。** Table 2 图注首句明确 "the 2D critic of Algorithm 1 on Game24 (first row) and variants with one element changed"。
8. **同意。** Figure 2 缩到 0.85 倍行宽，图高从 1.2 英寸加到 1.45 英寸给图内 legend 留位置。
9. **同意。** "The 2D critic keeps the update improving" 段先说 Figure 2 左图是什么（Table 1 两组 run 在 Game24 上的 held-out 轨迹），右图先解释交换设置（第四个 seed 在 250 步停下、换或不换 critic 继续），再说结论。
10. **同意。** 第二段开头先点出 §4.2 的两个设计元素（逐位置 head 及其在各位置读数处的斜率 eq. (12)、逐位置损失项 eq. (13)），再给"只能给上界"的结论和两个任务上各自需要哪一个。
9（重复编号，Table 3）. **同意（说明原因）。** 表注现写 "Variants are single runs unless marked (seed spread about a point, Appendix E.1.1); each reference is the three-seed mean of the unablated recipe its variant departs from"。原因是算力：消融是单次 1500 步运行，参照是已有三 seed 的运行；2D 的 seed 标准差约 1 分。若要补 seed，六个变体各两个 seed 约 120 GPU 小时，可另行安排。

## Section 6

1. **同意（校准三处）。** 结论按建议改为 "We propose CFRL ... in three variants" 开头。按事实改了三处：方法名用宏；"amortisation transfers its verified samples to cheaper samplers" 写成 "on two of the three tasks with an unlabelled pool"（Sudoku 没动）；scalar critic 除了 "flattens early" 还有 "on one task stays at the base"。CFRL-GRPO 增益小的原因句按 Appendix B.2 的新口径写成"其梯度分散在共享一个 reward bit 的噪声坐标路径上，测不到信号"，不再写成已证明的因果。新增的 limitation "CFRL-GRPO trained under the cheap chain" 采纳；"held-out checkpoint selection" 按事实改为对每个 RL 行都成立。

## Appendix B

1. **同意。** B.2 那句改为陈述测到的事实："the covariance of $r$ with a noise coordinate is, as far as we can measure, undetectable: the half-split cosine of the score-function gradient stays at the level of a zero-mean control up to $N=2048$ rollouts (Appendix E.2.3), while every coordinate contributes its full variance"，然后把 token 早期锁定写成 "a mechanism consistent with this, though not a proof of it"，并列出建议里的三条反例（置信度、注意力、自条件）。同段后文的 "few informative covariances" 和 "Table E.2.1 is flat" 也改掉（那张表并不平，写成"没有变体超过 +5"）。B.7 的统计量保留（便宜且有信息量），末句改为 "Most word decisions are therefore made in the first forward pass ... This statistic describes the decoded tokens, not the coordinates' influence on the reward"。§3 里 "takes its signal from few coordinates" 同步改为"方差随全部坐标增长，信号只来自坐标与 advantage 的协方差"。
2. **同 Section 3 第 0 条。** B.3 的 KL 讨论是关于梯度的数学陈述（KL 估计的梯度在 $\theta_{\rm ref}$ 处为零、之后不为零），与正文措辞无关，保留。

## 两轮 subagent 审查后额外修改的地方

- §2：$s_k$ 的定义改为"在第 $k$ 步形成、是该步调用的完整输入"（原写 "after step $k$" 不准确）；"Transition $k$" 的组成补上部分步；$\sigma_k$ 的作用改为噪声系数相加等于插值的噪声系数（原写法把信号系数归因给了 $\sigma_k$）。
- §3：无上标的 $\pi_{\theta_{\rm old}}$、$\pi_{\rm ref}$ 全部改为按因子的写法；协方差统一写 $I_d$。
- §4：Figure 1 图注 (b) 改为 "RL for diffusion and flow models"（原写 discrete diffusion，与引用的 Flow-GRPO 不符）；"Without the refit either critic is exploited" 的指向补上 Appendix E.3.7、E.3.4（Table 2 只画了 2D critic）；Algorithm 1 标题点名两个变体，"clipped" 写成 "gradient-norm clipped"；算法改为 `[H]` 放在引用处之后。
- §5：pick@$V$ 全文改为 pick@$V_\phi$（含 Appendix E）；Table 3 的源码移到消融段之后（该段在 PDF 里仍跨第 8、9 页，中间隔着 Table 2、Figure 2 和 Table 3，属正常换页）；loop 段用三个任务分别说明 coverage 的作用；"as the bias predicts" 改为 "leads one to expect"（两处）。
- Appendix B/D：Proposition 3 前定义 $\pi_\theta(w\mid c)$；Appendix D 的 scalar-critic 损失说明与 §4.2 统一为"只最小化第一项"。
- 第二轮审查后的修改：§3 里 θ_K 的定义补上"最后一步自己的 denoiser 调用"（$s_{K-1}$ 处的调用、带自身调用的确定性最后一步、解码），Appendix B 两处同步；Figure 2 图注注明交换实验里 2D head 只用 (13) 的第一项重拟合（Appendix E.3.6 的 recipe，在 Game24 上等价）；§4.3 注明预训练对两个 critic 都只用第一项；结论里 "keeps the policy improving" 限定为三个任务、"peaks above the base on every task at every seed"（Sudoku 的 2D seed 在 500 步后回落）；结论和 Appendix E.1.3 的 "cheap chain" 改为 "16-step chain $\gamma4$@16"（Sudoku 的便宜读出链是 $\gamma14$@16）；GPT-2 的 "same estimator" 补一句它用两步裁剪更新、flow 模型的两步版本在 Countdown 上增益不到两分；单步干预段用"区间不含零 / 区间到零"代替"well outside its interval"；"the copy is initialised from $\theta_{\rm ref}$"；距离符号 $D$（与 $D_\theta$ 冲突）改为 $\Lambda$（Table 1 图注、横轴、Appendix E.2.6）；Appendix B 的 $\omega_c$ 改 $\varpi_c$（$\omega$ 是 guidance 权重）、$\beta_0$ 改 $\chi_0$、协方差统一 $I_d$；Appendix E 的 $R_i$ 改 $R_{k,i}$、"denoising step" 改 "flow step"、Spearman 不再用 $\rho$；Figure 2 图例移到两个面板右下角空白处，纵轴下界降到 39.5。
- 未改：Appendix A 说 Sudoku 的便宜链是 $\gamma14$@16（这是 Table 4 和 loop 用的读出链），Appendix D 说 CFRL-GRPO 在所有任务上用 $\gamma4$@16 rollout；两者不矛盾，§5 的 setup 现在写明 CFRL-GRPO 在每个任务上都用 $\gamma4$@16 训练。
- 留给作者：Figure 1 图片内部面板 (b) 的标题仍写 "Discrete diffusion places reward along denoising step axis"，与图注新写法（diffusion and flow models）不一致，且图内字号约 4.7pt；需在 Keynote 源文件里改。
