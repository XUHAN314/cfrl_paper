# 对 experiment_suggestions.md 的逐条回复（2026-09-23，第二版）

说明：逐条对应 `experiment_suggestions.md` 的编号。每条给"同意 / 不同意 / 部分同意"。同意的写 main 里怎么改；不同意的给理由。改动只在 `main.tex` 的输入文件（`sections/*.tex` 不含 `*1.tex`、`*2.tex`，`appendix/*.tex`），Abstract 和 Intro 没动。页数不作为约束：正文现在到第 10 页（Algorithm 1、§4.4 和 §5 开头在第 7 页，Table 1 在第 8 页顶，Table 3、Table 4 和 Conclusion 在第 10 页，参考文献从第 11 页开始）。改完后过了两轮 subagent 审查（改动处的逻辑、全文记号一致、行文连贯），审查发现的问题一并修了，列在文末。

一个先说的事实（与 §5 第 1 条有关）：Table 1 里 GPT-2 + token GRPO/PPO 的数字（53.9 / 1.4 / 93.2 / 32.2）来自 9 月 8 到 15 日 dropout 开着的运行；9 月 15 日还跑过 dropout 关掉（PPO 的标准做法，脚本现在的默认）的同款运行，Countdown、Game24、Sudoku 更高（61.7、46.2、1.6），K&K 持平（93.2），同一评测。这组结果一直没进论文。两种 recipe 我各补了两个 seed（见 §5 第 1 条），放哪一组、要不要换，等作者核对后决定，本轮不改 Table 1。

## Section 2

1. **同意。** "time grid" 改为 "time steps $0=t_0<\dots<t_K=1$"。
2. **同意。** "steps $\Delta_k$" 改为 "step lengths $\Delta_k=t_k-t_{k-1}$"。
3. **不同意完全删除，做法是细节进附录。** CFG 的公式、权重 $\omega=2$、"critic 的拷贝不用 guidance" 移到 Appendix D 的模型描述段；正文留半句 "A call may combine a prompt-conditioned and an unconditioned evaluation (classifier-free guidance, Appendix D); we treat the combination as one denoiser call"。理由：§4.2 说 critic 的拷贝 "evaluated without guidance"，Appendix C 的显存论证依赖"一次调用是两次评估"，正文不能对这个词零解释。
4. **同意。** 改为 "between scaled Gaussian noise $\varepsilon\sim\mathcal N(0,I)$ and the clean embeddings $x$, for $t\in[0,1]$ and a fixed noise scale $c_\sigma$"。
5. **同意。** 在写 $y_0$ 之前先引入 $y_k$："adds fresh noise to the state before every denoiser call (\emph{re-noising}); we write $y_k$ for the re-noised embeddings. Starting from noise, $y_0=c_\sigma\varepsilon_0$ ..."。其余结构不动（见第 12 条）。
6. **不同意换词。** "re-noise/re-noising" 是扩散采样文献的常用说法，也是 ELF 自己的操作名。只在首次出现处用斜体定义一次，之后照用。
7. **同意。** 三行更新式之后紧跟四个常数的定义，再逐个解释作用：$\alpha_k$ 是保留的上一状态比例，它把信号系数从 $t_{k-1}$ 降到 $\alpha_kt_{k-1}=\tilde t_k$，denoiser 因此在 $\tilde t_k$ 被查询；$\sigma_k$ 是注入噪声的尺度，使噪声系数加起来等于插值在 $\tilde t_k$ 的噪声系数（$\alpha_k(1-t_{k-1})+(1-\alpha_k)=1-\tilde t_k$，ELF 的约定）；$\lambda_k$ 是部分步从 $y_k$ 走向估计的比例，走完后状态到达 $t_k$。
8. **不同意照 main2 改写。** 我读了 main2 的 §2。它把三行更新式压成一行、四个常数塞进一句，比现在更难读；它的"读者去看原文"问题在我们这版里靛第 5、7 条解决。没有采纳它的任何结构。
9. **同意加 Bernoulli 说明，不同意 main2 的条件对象。** eq. (4) 写成 $V^\star_k(s)=\mathbb P(r=1\mid s_k=s,c)$ 并注明 $r\mid s_k\sim\mathrm{Bernoulli}(V^\star_k(s_k))$，把"给定中间状态时 reward 是随机变量"写明。main2 把 value 条件在 $y_k$ 上，不对：马尔可夫状态是 $(y_k,\hat x_{k-1})$，只给 $y_k$ 不够决定后续分布（自条件输入 $\hat x_{k-1}$ 也进入 denoiser）。我们保持原来的写法。
10. **不同意删除。** 这一段是全文的概念支点：两个梯度恒等式的对比、"AR 的每一项是类别决策"（flow 的对应说法"每一项是噪声坐标"在 §3 eq. (7) 之后展开）、以及 Appendix B.4 用到的正则性条件都在这里，§4 和 Appendix B 都指回 §2。删掉后这些内容要在 §4.1 重写，篇幅不省而连贯性受损。只做了压缩：保留两个恒等式、AR 对比和"reward 没有可微路径、中间状态的 value 有"。
11. **同意。** 真值统一写为 $V^\star_k$（大写星号，与 $V_\phi$ 对应），eq. (4) 后注明 "the star marks the true value; the critic $V_\phi$ of §4 will approximate its log-odds $\mathrm{logit}\,V^\star_k$"。全文替换：§2、§4、Appendix B（约 50 处）、Appendix C（17 处），包括 $V^\star_{K-1}$、$V^\star_{k\pm1}$、$V^\star_1$、$\nabla_yV^\star_k$、$\mathrm{logit}\,V^\star_k$。
12. **不同意。** 不把 MDP 改到 $s$ 上、不把全文改成只用 $s$。$s_k:=(y_k,\hat x_{k-1})$ 在我们原稿里就只是公式里的简写（在 partial step 那行定义，同段后文说明它是 denoiser 调用的完整输入、链在 $s_k$ 上是马尔可夫的），这个写法保持原样，没有再往 $s$ 上重构。理由：(a) 论文的核心机制是 eq. (8) 的两个通道，一个经再加噪状态 $y_k$、一个经估计 $\hat x_{k-1}$，"第二通道占大部分梯度"、score-function 只作用在 $y_k$ 上、critic 消融"去掉 $\hat x_{k-1}$ 输入"这些论点都必须用分量写，只用 $s$ 反而说不清；(b) §3 的高斯因子是 $y_k$ 给定 $s_{k-1}$ 的密度，eq. (6)、(7) 离不开 $y_k$；(c) 把全文的 $y$、$\hat x$ 换成 $s$ 的分量记号，只是换符号，不增加任何信息。所以这条以及 main2 里所有围绕 $s$ 的写法一律不采纳。

## Section 3

0. **同意。** 删掉 "the anchor protects nothing" 那句。现在写 "$\beta=0$: as in group-relative REINFORCE (Yao et al., 2026) and DAPO (Yu et al., 2025), the policy is not anchored to the initial model. Our programmatic verifiers leave little to exploit, and on the two tasks where we tried it, adding the anchor to the clipped two-step form undid its gain and left it no better than CFRL-GRPO (Appendix E.2.2)." 最后半句是 Appendix E.2.2 表格的高层结论，不带数字。引用核实：Yao et al. 是 ICLR 2026 正式论文，我下载了 PDF，正文明确说他们的 GRPO 实验不用对参考模型的 KL 正则并引 DAPO 说这类项常不必要，可以引；DAPO 本来就在 bib 里，它明确去掉 KL 项。Proposition 1 的证明不需要引用。
1. **同意。** 现在写 "We write $u$ for a factor of the rollout density, a transition $k$ or a token $i$, and $\pi^u_\theta$ for that factor"，两类因子显式定义为 $\pi^k_\theta:=\prod_{i\in\mathcal G}\mathcal N(\alpha_kD_\theta(s_{k-1})_i,\sigma_k^2I_d)$ 和 $\pi^i_\theta:=\mathrm{softmax}(\ell_i)$（$\ell_i$ 是 $z_K$ 处的解码 logits）；参考因子 $\pi^u_{\rm ref}:=\pi^u_{\theta_{\rm ref}}$ 也在此定义，§3 不再出现无上标的 $\pi_\theta$、$\pi_{\rm ref}$、$\pi_{\theta_{\rm old}}$。$\pi^{\rm AR}$ 在 §2 只用于 AR 模型，带上标，不冲突。
2. **同意。** eq. (6) 改用 $\log\pi^k_\theta(y^g_k\mid s^g_{k-1})$ 和 $\log\pi^i_\theta(w^g_i\mid z^g_K)$，不再写出高斯。eq. (7) 处保留高斯展开，因为 "score = 噪声/σ 乘 Jacobian" 要从密度直接看出来，这是 §3 的论点。
2.5. **不同意。** 与作者 9 月 21 日定的规则冲突："必须先给原始 GRPO 的形式，不能直接跳到我们的版本，不能上来就一个变体版本然后 claim 这是 GRPO"。eq. (5) 是原始 GRPO 目标，是我们版本的出发点，Proposition 1 证明的就是从 (5) 到 (6) 的等价。(6) 已按第 2 条写紧凑。
3. **同意。** 不再用一个 $\nu_u$ 两个值：正文写 "a random subset $\mathcal K$ of the transitions, with weight $\nu_k=1$, and all tokens, with weight $\nu_i=1/|\mathcal G|$, so that the readout counts as one transition"；eq. (6) 本来就是两个显式求和。eq. (5) 保留 $\sum_u\nu_u$ 的一般形式，因为它是标准 GRPO 的写法，$u$ 和 $\nu$ 已在前文定义。
4. **同意。** 删掉 $\delta$，advantage 写成 $\frac{r^g-\mathrm{mean}(r^{1:G})}{\mathrm{std}(r^{1:G})}$；Appendix B.4 关于 std 归一化的备注同步去掉 $\delta$。
5. **部分同意。** 采用了"两个改动"的结构（"CFRL-GRPO makes two changes to (5)"，主语用方法名与全文一致；不写 "flow-specific"，因为 $\beta=0$ 借自 DAPO 不是 flow 特有）、"one on-policy update per batch"、"still depends on $\theta$"。不采纳两处：(a) "providing little benefit" 与 Appendix E.2.2 矛盾（逐位置比值加第二次更新在两个任务上有增益），改为 "the log-ratio of a whole Gaussian transition sums over $|\mathcal G|\,d$ coordinates, so the ratio leaves the clipping interval after the slightest change of $\theta$, and with per-position ratios what a second step buys is task-dependent (Appendix E.2.2)"；(b) 保留 "ratio and clipping matter only from the second step on (Proposition 1)"，这是与附录证明的衔接。
6. **同意。** eq. (7) 和比值都改成 `\frac{}{}`；比值按因子写（转移因子的比值是 $\pi^k_\theta(y^g_k\mid s^g_{k-1})/\pi^k_{\theta_{\rm old}}(y^g_k\mid s^g_{k-1})$），与 eq. (6) 的条件形式一致。
7. **不同意（暂不引用）。** openreview 链接被浏览器验证页拦住，arXiv 搜不到同名论文，无法核实这篇 workshop 论文的内容；未核实的文献不引。"on-policy 时裁剪不激活"是比值恒为 1 的直接推论，Proposition 1 已证明，不需要外部引用。bib 条目 `zhang2026importance` 保留，coauthor 若确认原文内容，可加一句 "see also"。

## Section 4

1. **不同意（按作者指示）。** Figure 1 不放大。另注：coauthor 已在提交 91de7fe 里更新了图，面板 (c) 现在是 "CFRL-SVG-2D"。
2. **同意。** 开头改为：先说 CFRL-GRPO 无需 critic 但付出噪声因子的代价，"We now introduce two actor-critic variants based on stochastic value gradients (SVG; Heess et al., 2015) that differentiate the chain itself and so remove that factor"（SVG 在 §4 首段首次展开，节标题已用缩写），然后是 reward 无可微路径、中间状态的 value 有、两个变体都学 log-odds 的 critic 并经一个转移上升。保留了作者 21 日批准的 SVG(1) 结构句，最后一句说两个变体在 critic 分辨率上不同。
3. **部分同意。** §4.1 开头加了段名 `\paragraph{Transition-wise decomposition.}` 和目的句 "We first split $\nabla_\theta J$ into one term per transition, then turn the term at one transition into an estimator with a critic"。不采纳建议的正文，因为其中 "$\theta_K$ parameterizes the final deterministic step and decoder" 不对：$\theta_K$ 还包括 $s_{K-1}$ 处的调用和最后一步自己的调用，正文写 "the copy used from $s_{K-1}$ on: the call at $s_{K-1}$, the deterministic final step with its own call, and the decode"。
4. **同意。** eq. (8) 去掉挤占宽度的范围标注（前一句已说明 $k\le K-2$），公式号不再溢出。
5. **同意。** §4.2 首句改为 "We now design the critic. It reads what the value is a function of, ..."。
6. **同意。** eq. (11) 换序，两个变体都以 $V_\phi(s_k)=$ 开头。
7. **同意。** Require 改为 "policy parameters $\theta$; pre-trained critic $V_\phi$ with trainable head parameters $\phi$ on a fixed copy of the denoiser; verifier $r$; sampler $(\gamma,K)$; window $\mathcal W$; buffer $\mathcal B$ and its size $N_{\mathcal B}$; iterations $T$, prompts per iteration $B$, critic minibatch $B'$ and steps $m_{\rm c}$, AUC check every $n_q$ iterations on the newest $N_q$ states with floor $a_0$ and $m'_{\rm c}$ extra steps; learning rates $\eta_\theta,\eta_\phi$"。正文两处 rollout 标明 "under $(\gamma,K)$"，buffer 行写 "keeping the newest $N_{\mathcal B}$"，批下标用 $s^g_k$（与 §3 的 rollout 下标一致）。Require 里补了 "window $\mathcal W$ with top step $k_{\rm hi}$"；正文用到的符号除 eq. (13) 定义的 $\mathcal L_{\rm 2D}$ 外都在 Require 里。
8. **同意。** 改为 "the 2D critic minimises (13), whereas the scalar critic minimises the first term alone. The first term fits $V_\phi$ to $\mathrm{logit}\,V^\star_k$ ...; the second, present only for the 2D critic, makes every reading a calibrated log-odds on its own"。相应地，§4.2 和 §5 里"两个 critic 只在 head 与均值的顺序上不同"改为"在顺序上不同，并由此在损失上不同"。
9. **部分同意。** 采用 "We train the critic jointly with the policy because ..." 的开头，AUC 用"对新状态结果的排序质量"解释（全称在 Appendix D）。不采纳三处：(a) 保留 bias $b_k$ 这条线（"two things move the condition under which the bias vanishes"），它把 §4.3 与 eq. (9)、Appendix B Step 5 连起来；(b) 建议文本 "Both networks are trained under the sampler that will be deployed" 漏了 Countdown critic 预训练用便宜链的例外，正文保留例外的指向；(c) "The copy stays frozen" 指向 Table 2 而不是实验小节。另外原来的 "Amortising the policy into the model" 从 §4.3 独立为 §4.4，因为它不属于"训练 critic"。

## Section 5

0. **同意。** 开头改为 "We provide empirical support for each claim of §3–§4 through a corresponding table or figure"。
1. **同意补 GPT-2 的 seed，不同意补 LLaDA 的 seed，结果不进论文。** 已提交 GPT-2 PPO 在四个任务上各两个新 seed（seed 1、2；原来是 seed 0），并且两种 recipe 都跑：论文现用的 dropout 开着的版本，和脚本现在默认的 dropout 关掉的版本。作业号在 `flow-nlp-rl/slurm/jobmaps/gpt2_seeds_jobs.txt`，结果汇总到 notes，等核对后再决定 Table 1 放哪一组以及是否写成三 seed 均值。LLaDA-8B 一次评测要大半天、训练一天多，单 seed 加表注说明即可。
2. **同意。** Table 1 的三个 ELF-B RL 行每格加了相对 base 的增益（小字括号，例如 Game24 的 "58.8 (+13.8)"，12 个数均已核对）；表改为 `\footnotesize`、左栏加宽到 0.615，右侧面板缩到 0.37。
3. **部分同意。** Figure 2 两个面板改为图内 legend（左：scalar critic、2D critic、base；右：四条交换曲线），图注删掉线型键。Table 1 和 Table 2 右侧的面板在页面上只有约 2 英寸宽，图内 legend 会盖住数据，这两处保留图注里的线型键。
4. **同意压缩图注，不同意删掉 $D$。** Table 1 图注从 10 行压到约 7 到 8 行（右侧面板只说测什么、怎么读，定义放正文和 Appendix E.2.3、E.2.6）；Table 2 图注压缩并写明是 2D critic；Figure 2 图注压到 3 到 4 行；Table 3、4 各压一两行。步长的参照距离保留：横轴用它的分数才读得出"四分之一到一半的距离就 +4"，换成绝对步长没有意义；图注用一句定义它。审查指出 $D$ 与部分步映射 $D_\theta$ 撞符号，改名为 $\Lambda$（图注、横轴、Appendix E.2.6 同步）。
5. **同意。** 拆成 "Main result." 和 "Why the score-function variant is capped." 两段，后者带 `sec:exp_estimator` 标签，以"估计量的各种变体在个别任务上能多几分，但在任何任务上都到不了 CFRL-SVG-2D 的水平；§3 把上限归因于动作空间"开头作为论点。
6. **同意。** 右侧面板的段落先说设置再说结论：half-split cosine 是把 $N$ 条 rollout 分两半、取两个均值梯度的余弦，真信号随 $N$ 增大趋向 1、纯噪声停在 0 附近；单步干预是从 base 沿各均值梯度走一步 AdamW 归一化的步长再评测，检验方向；结论用"区间不含零 / 区间到零"说，"raw gradient" 写成 "the same pathwise gradient without AdamW's per-coordinate normalisation"。
7. **同意。** Table 2 图注首句写明 "the 2D critic of Algorithm 1 on Game24 (first row) and variants with one element changed"。
8. **同意。** Figure 2 缩到 0.85 倍行宽，图高从 1.2 英寸加到 1.45 英寸给图内 legend 留位置，纵轴下界降到 39.5 让图例落在空白处。
9. **同意。** "The 2D critic keeps the update improving" 段先说 Figure 2 左图是什么（Table 1 两组 run 在 Game24 上的 held-out 轨迹），右图先解释交换设置（第四个 seed 在 250 步停下、换或不换 critic 继续），再说结论；"2D head 在这四个 run 里只用 (13) 的第一项重拟合" 写在图注里（Appendix E.3.6）。
10. **同意。** 第二段开头先点出 §4.2 的两个设计元素（逐位置 head 及其在各位置读数处的斜率 eq. (12)、逐位置损失项 eq. (13)），再给"只能给上界"的结论和两个任务上各自需要哪一个。
9（重复编号，Table 3）. **同意说明原因，不补 seed。** 表注现写 "Variants are single runs unless marked (seed spread about a point, Appendix E.1.1); each reference is the three-seed mean of the unablated recipe its variant departs from"。消融是单次 1500 步运行，参照是已有三 seed 的运行；2D 的 seed 标准差约 1 分；消融在 best 处差 1 到 9 分、在 final 处差 2.5 到 28 分（两个学习率变体的差距体现在 final 处）。补 seed 要六个变体各两个、约 120 GPU 小时，收益小，没有提交。

## Section 6

1. **同意，校准三处。** 结论按建议改为 "We propose CFRL ... in three variants" 开头。按事实改：方法名用宏；"amortisation transfers its verified samples to cheaper samplers" 写成 "on two of the three tasks with an unlabelled pool"（Sudoku 没动）；scalar critic 除了 "flattens early" 还有 "on one task stays at the base"；2D critic 写成 "keeps the policy improving through the joint refit on three tasks and peaks above the base on every task at every seed"（Sudoku 的 2D seed 在 500 步后回落）。CFRL-GRPO 增益小的原因句按 Appendix B.2 的口径写成"其梯度分散在共享一个 reward bit 的噪声坐标路径上，在 base 的 rollout 上与零均值对照一样不自洽"，不写成已证明的因果。新增的 limitation 采纳，写成 "CFRL-GRPO trained under the 16-step chain $\gamma4$@16 rather than the deployment chain"（Sudoku 的便宜读出链是 $\gamma14$@16，不能笼统写 cheap chain）；"held-out checkpoint selection" 对每个 RL 行都成立，不只 SVG。

## Appendix B

1. **同意。** B.2 那句改为陈述测到的事实："the covariance of $r$ with a noise coordinate is, as far as we can measure, undetectable: the half-split cosine of the score-function gradient stays at the level of a zero-mean control up to $N=2048$ rollouts (Appendix E.2.3), while every coordinate contributes its full variance"，然后把 token 早期锁定写成 "a mechanism consistent with this, though not a proof of it"，并列出建议里的三条反例（置信度、注意力、自条件）。同段后文的 "few informative covariances" 和 "Table E.2.1 is flat" 也改掉（那张表并不平，写成"没有变体超过 +5"）。B.7 的统计量保留（便宜且有信息量），末句改为 "Most word decisions are therefore made in the first forward pass ... This statistic describes the decoded tokens, not the coordinates' influence on the reward"。§3 里 "takes its signal from few coordinates" 同步改为"方差随全部坐标增长，无论它们是否带信号"。
2. **同 Section 3 第 0 条。** B.3 的 KL 讨论是关于梯度的数学陈述（KL 估计的梯度在 $\theta_{\rm ref}$ 处为零、之后不为零），与正文措辞无关，保留。

## 两轮 subagent 审查后额外修改的地方

- §2：$\sigma_k$ 的作用原先写成"使信号系数匹配"，错，改为噪声系数相加等于插值的噪声系数；"Transition $k$" 的组成补上部分步。
- §3：无上标的 $\pi_{\theta_{\rm old}}$、$\pi_{\rm ref}$ 改为按因子的写法；协方差统一 $I_d$；"比值对坐标求和"改为"对数比值求和"。
- §4：$\theta_K$ 的定义补上最后一步自己的 denoiser 调用（Appendix B 两处同步）；"两个更新接近"的指向改为 Appendix E.3.4、E.3.5；Figure 1 图注 (b) 改为 "RL for diffusion and flow models"（原写 discrete diffusion，与引用的 Flow-GRPO 不符）；"Without the refit either critic is exploited" 补指 Appendix E.3.7、E.3.4（Table 2 只画了 2D critic）；§4.3 注明预训练对两个 critic 都只用 (13) 的第一项；Algorithm 1 标题点名两个变体，"clipped" 写成 "gradient-norm clipped"，算法用 `[H]` 放在引用处之后；"initialised from $\theta_{\rm ref}$"。
- §5：GPT-2 的 "same estimator" 补一句它用两步裁剪更新、flow 模型的两步版本在 Countdown 上增益不到两分；pick@$V$ 全文改为 pick@$V_\phi$（含 Appendix E）；Table 3 的源码移到消融段之后；loop 段用三个任务分别说明 coverage 的作用；"as the bias predicts" 改为 "leads one to expect"（两处）；setup 写明 CFRL-GRPO 在每个任务上都用 $\gamma4$@16 训练。
- Appendix B/D/E：Proposition 3 前定义 $\pi_\theta(w\mid c)$；Appendix D 的 scalar-critic 损失说明与 §4.2 统一为"只最小化第一项"；E.2.2 里 anchor 的效果改为"把两步版本拉回 base 上方约一分，不比 CFRL-GRPO 好"；E.1.3 的 "cheap $\gamma4$@16 chain" 改为 "16-step $\gamma4$@16 chain (the cheap chain except on Sudoku)"；$\omega_c$ 改 $\varpi_c$（$\omega$ 是 guidance 权重）、$\beta_0$ 改 $\chi_0$；$R_i$ 改 $R_{k,i}$、"denoising step" 改 "flow step"、Spearman 不再用 $\rho$。

## 第二轮审查后的修改

- §2：$\sigma_k$ 的说明补上"两个噪声独立，匹配的是系数不是方差"（ELF 的约定是系数线性相加）；"The display groups the operations by step; the density groups them by transition" 一句衔接步与转移的两种分组；删掉与首段重复的 prefix 句；注明 $z_k$ 是时刻 $t_k$ 的状态；两个估计量段里 "it" 的指代和"value 的可微路径依赖正则性假设"写明。
- §3：$\pi^k_\theta$ 写成逐位置高斯的乘积（原来的 $I_d$ 维度不对）；$\pi^i_\theta:=\mathrm{softmax}(\ell_i)$ 不再经过 $p_\theta$；Appendix B 里整条转移的高斯协方差改回 $I$。
- §4：critic 拷贝那句拆短；§4.4 首句改为 "The gain of the SVG variants is made on their training chain, and a cheaper chain through the same denoiser need not inherit it"（原句"属于训练链"与 E.4.2 的 K&K 结果矛盾）。
- §5：setup 里定义 $\gamma56$@64 记号（churn 与步数）；单步干预的结论按表格改写：Λ/16 以内两种步都只动约一分，Λ/4 到 Λ/2 时 pathwise 的区间不含零、score-function 增益更小且区间到零（原句在小步长处与表格矛盾，Appendix E.2.6 同步）；"refit at $k_{\rm hi}$ matters" 限定为 Countdown、在 Game24 上中性；"per rollout" 补进坐标计数。
- §6：CFRL-GRPO 那句改为"梯度分散在共享一个 reward bit 的噪声坐标路径上，在 base 的 rollout 上与零均值对照一样不自洽"（原来的"测不到任何信号"与 GRPO 自己的 +3.5/+4.1 矛盾）；scalar critic 写成 "flattens or turns back early"（Countdown 是回落不是变平）。
- Appendix：Λ 的两处漏改（E.2.6）；Spearman 不再用 $\rho$；"cosine schedule of Algorithm 1" 改为 Appendix D（算法里 $\eta_\theta$ 是常数符号）；bare $V$ 改 $V_\phi$；E.3.3 提到 Countdown/K&K 时指向 Figure 5；loop 的轮次参数改上标 $\theta^{(\varrho)}$（与转移副本 $\theta_k$ 区分）；Appendix A 的 $\gamma g$@$K$ 写法去掉与 rollout 下标冲突的 $g$。
- Figure 2：交换时刻的竖虚线只画到 45.5 以上，不再穿过图例。

## 第二轮审查后的修改（续）

- §2 和 §4.1：高斯平滑的表述改为与 Appendix B.4 一致，只在再加噪的自变量上平滑（中心在 $\alpha_kD_\theta(s)$），估计通道原样传递，不再写成"读在 denoiser 输出处的平滑"。
- §4.2："汇合场的上升要由 $|\mathcal G|$ 个读数承担"那句不再把这一点归到第二项（Game24 上没有第二项增益也在），改为"每个读数由 head 在自己位置上读出，加上第二项后各自拟合结果"。
- §4.4 与 Appendix B.6、C.3：Proposition 3 里 $\pi_\theta(w\mid c)$ 的定义改为"抽取该轮样本的链、普通采样"；引导采样（剪枝加推动）的轮次另有一个成功集合上的分布 $q'$，bound 带同样的 KL 松弛；正文写 "tight at the current model for plain sampling before de-duplication"；便宜链上 critic 排序的说法改为"训练论证不覆盖它（在 Countdown 上测过，Appendix E.4.1）"。
- Appendix A 与 D 关于 K&K 的 pool：A 原写"loop 的 pool 与训练集不相交"，D 写"第一轮用 50k 训练题"，改 A 为"第一轮用不带标签的训练题，之后每轮用与训练、验证互不相交的新 50k"；§5 的 loop 段加半句说明。
- §5：2D-critic 段的曲线描述按 E.3.3 改（scalar 在约 500 步后变平、2D 继续上升；K&K 同形，Countdown 的 scalar 回落，Sudoku 两者 500 步后都回落）；单步干预的 score-function 一句写成"Game24 上不到两分、Countdown 上没有，区间含零"；Λ 的定义改为"2D-critic run 从 base 走过的参数距离（Game24 上 750 步）"；"mean beats minimum" 指向 Appendix C.3 解释为什么搜索用最小值。
- Appendix D：loop 每题采样数原写 "4 seeds / 8 seeds / 2 seeds"，改为 $n{=}4/8/2$ samples per problem（seeds 在全文另有含义）。Appendix E：E.2.6 的 score-function 一句按表重写、"destroys the policy at a quarter to a half of Λ"；E.2.3 的 "sixteen times its batch carries no more signal" 写具体；E.3.6 的 57.3 与参照的 final 58.0 比（原与 best 58.8 比）；E.3.10 补 "Game24 上五步 buffer 与单步持平"；两处指向和"predicts"改正；tab:samplers 的 dagger 说明改到正确的列上。B.1 的读出写成 $\prod_i\pi^i_\theta$。

## 留给作者

- Table 1 的 GPT-2 PPO 数字用 dropout 开还是关的运行（见开头），以及要不要换成三 seed 均值；新 seed 的结果在 notes 里等核对。
- Figure 1 图片内部面板 (b) 的标题仍写 "Discrete diffusion places reward along denoising step axis"，与图注新写法不一致，图内字号约 4.7pt；需在 Keynote 源文件里改。
- `zhang2026importance` 是否引用，等 coauthor 确认原文内容。
