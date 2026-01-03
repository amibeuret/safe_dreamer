---
layout: post
title: Learning safety guardrails for surgical robotics
date: 2025-12-30
permalink: /learning-safety-guardrails/
---

<p style="color: #666; font-style: italic; margin-bottom: 2em;">{{ site.author }}<br/>{{ page.date | date: "%B %Y" }}</p>

Robotics, and even more so, surgical robotics, are settings where the drawbacks of usual reinforcement learning are pronounced. 

We cannot afford to learn by trial and error. Mistakes could be costly or even life threatening. Even if we train our models in high-fidelity simulations, it remains quite sample inefficient to train safe policies. Unlike games or simulated environments, reward or cost signals might not be even clearly defined. 

What we have, in many hospitals, is a growing pile of recorded operations: videos, robot states, actions, and outcomes. Mostly safe; Mostly unlabeled; Rarely optimal, and often partially observed and high-dimensional such as RGB or ultrasound images.

This post is about what to do with this reality. Can we learn _safe_ policies from such offline data with minimal annotation and interaction? Can we _infer_ safety constraints that generalize beyond the demonstrated behavior?<sup id="footnote1-ref">[[1](#footnote1)]</sup> 

This post is less about the technical details (which are in the linked papers) and more about the overall design choices and reflections on what works and what does not. It might help you avoid some pitfalls. If you don't have enough time, **jump to [the discussion](#discussion-and-takeaways)** or **the [future work](#future-work)** at the end, the rest is details.


First I describe a simple offline-first pipeline briefly and explain why some design choices were made to tackle the issues above. Then in the discussion section I share more opinions and personal experience on the topic. 

The pipeline has two complementary ways to _learn_ constraints from demonstrations:

- A geometric approach that learns a conservative safe region as a convex polytope approximation in a learned latent space.
- A preference learning approach that learns a conservative cost from a small number of pairwise preference labels, without requiring unsafe trajectories.

Then using these learned constraints (from either of these two approaches), we can enforce safety during model-based planning to learn safe policies, either offline, or online with minimal interaction.

<figure>
  <img src="{{ site.baseurl }}/assets/videos/sono-preference_learning.gif" alt="Example of preference-based constraint learning on ultrasound navigation." style="width:100%; max-width:900px;">
  <figcaption>
    Constraint learning applied to ultrasound probe navigation: learning safe behavior from expert preferences.
  </figcaption>
</figure>

## The problem formulation

A useful formalization of surgical procedures is a constrained Markov decision process (CMDP)&nbsp;<span id="ref1-ref">[[1](#ref1)]</span>:

- We still want to maximize task return (reach a target, orient an instrument, track a path, complete a subtask).
- But we must keep expected cumulative cost below a budget.

We say a policy is safe if the expected cost is below the budget.<sup id="footnote2-ref">[[2](#footnote2)]</sup> 

This CMDP framing matters because it separates two jobs that are often conflated:

- Learning how to do the task.
- Learning what we are not allowed to do.

We are interested not only in learning safe policies, but also in _inferring_ safety constraints from data. This is important because in many surgical settings, we do not have explicit cost functions or safety constraints defined upfront. Instead, we have demonstrations of safe behavior, and we want to extract constraints that can guide future decision making.

## The pipeline

<figure>
  <img src="{{ site.baseurl }}/assets/images/unified_framework_pipeline.png" alt="Unified pipeline: offline feature learning, offline constraint learning, and constrained model-based planning." style="width:100%; max-width:1100px;">
  <figcaption>
    A very simple, yet effective pipeline for safe policy optimization: 1. offline feature learning, 2. offline constraint learning, then 3. constrained model-based planning and policy optimization.
  </figcaption>
</figure>

The pipeline has three components. Both constraint-learning approaches plug into the same middle block.

### 1) Offline feature learning for partial observability

Real surgical sensing is partial. We see images, not necessarily full state. A single frame is not enough; the belief over what is happening matters <span id="ref2-ref">[[2](#ref2)]</span>.

So the first step is to learn a latent state that summarizes history. If the learned latent is a sufficient statistic of history, approximating the true Bayesian belief state, then the planning in latent space approximates planning in the corresponding belief MDP, rendering the problem effectively Markovian: the better the approximation, the closer we get to an MDP. 

Concretely, we use a recurrent state-space model (RSSM) <span id="ref3-ref">[[3](#ref3)]</span> trained from recorded trajectories. It learns to compress high-dimensional observations into a latent state and to predict forward under actions. It is intended to approximate the true belief state by capturing history dependence in its recurrent state. 

Although it has no guarantees, empirically we observe that the RSSM trained purely for reconstruction and dynamics prediction can already encode safety-relevant structure in its latent space. It does not need gradients from a cost signal. This makes it practically a self-supervised approach with respect to safety signals. When additional supervision is available, it can sharpen further, but the baseline is already surprisingly strong.

By learning this latent space, not only we handle partial observability for the sake of task performance, we also reduce the required number of safety labels significantly down the road. Instead of learning constraints in pixel space, we learn them in this rich latent space. 

I can only recommend this step strongly: it is the foundation that makes any kind of downstream learning more sample efficient and robust.

### 2) Offline constraint learning in the latent space

Given a learned latent state, we want a constraint model that can be enforced during decision making. We learn this model offline from demonstrations and a small amount of annotation, depending on which route we choose:

- [**Polytope learning**](#option-a-learning-a-conservative-fence-with-convex-polytopes): learn a conservative safe set from labeled safe and unsafe trajectories.
- [**Pairwise comparisons**](#option-b-learning-constraints-from-comparisons): learn a conservative cost from pairwise comparisons, without requiring unsafe trajectories.

We will cover both in detail below.

### 3) Constrained model-based planning and policy optimization

Once we have a world model and a constraint model, we can keep interactions minimal:

- Roll out imagined trajectories in latent space using our world model (dreaming).
- Optimize actor and critic against predicted reward while respecting constraints.
- If we go online at all, it is in a controlled, model-based way that reduces the number of real interactions significantly. It also enables many existing safe planning or exploration methods to be plugged in.

If safety and sample efficiency matter, model-based planning is the way to go. Model-based planning enables us to evaluate candidate policies in imagination or tree search, reducing real-world interactions to a minimum. It also enables safer exploration strategies.

That's it. That is the pipeline. If you are curious about the constraint learning options, read on. Otherwise, jump to the [discussion](#discussion-and-takeaways) section for high-level takeaways.

## Constraint learning methods

Now let's dive into the two constraint learning options.

### Option A: learning a conservative fence with convex polytopes

We know that the convex hull of safe trajectories defines a safe set<sup id="footnote3-ref">[[3](#footnote3)]</sup>: if we stay inside, we are safe. If we step outside, we might be unsafe <span id="ref4-ref">[[4](#ref4)]</span>. This approach is a conservative one: it may label some safe points as unsafe, but it will not label unsafe points as safe. Exactly what we want for our high-stakes setting.

With this approach, not only we get nice theoretical properties, we also handle the usual challenge of _heterogeneous_ demonstrations: different surgeons, different styles, different situations. This is exactly where IRL methods struggle because they try to learn a _single_ reward function that explains all behavior. Here however, we only care about the boundary between safe and unsafe. Compared to reward learning, constraint learning can be less sensitive to stylistic variation. Another advantage is that we can relax the _optimality_ assumptions usually needed for IRL.<sup id="footnote4-ref1">[[4](#footnote4)]</sup> This is a big deal in practical settings where demonstrations are rarely optimal.

Now computing the convex hull in high dimension is practically intractable. For this, we approximate it with a K-facet polytope using a Convex Polytope Machine (CPM) <span id="ref5-ref">[[5](#ref5)]</span>. We lose the cherished guarantees of the original convex hull<sup id="footnote5-ref">[[5](#footnote5)]</sup>, but we gain tractability, scalability, and additional way of controlling conservatism via a margin parameter. 

CPM learns a set of hyperplanes that define a convex region. Safe points are pushed to lie inside; unsafe points are pushed out. The result is an approximation of the safe set that is tractable to learn and enforce.

<figure>
  <img src="{{ site.baseurl }}/assets/videos/cpm-construction.gif" alt="Animation showing CPM learning a polytope boundary around safe trajectories." style="width:100%; max-width:900px;">
  <figcaption>
    CPM in action: learning a convex polytope that tightly bounds safe behavior while excluding unsafe trajectories. Green points are safe policies, red points are unsafe policies, blue lines define the exact polytope, and the yellow lines define the hyperplanes learned by CPM.
  </figcaption>
</figure>

Once we have K facets, each facet becomes a linear constraint that can be integrated into model-based planning. In practice, this is done by translating the polytope inequalities into budgeted constraint terms and optimizing with constrained objectives (for example via Lagrangian style updates).

### Option B: learning constraints from comparisons

Sometimes we cannot label costs. Sometimes we cannot even obtain unsafe examples. But we can often ask a surgeon a simpler question:

```
Which of these two trajectories is safer (or more acceptable)?
```

That is the core of the preference-learning option <span id="ref6-ref">[[6](#ref6)]</span>. It uses a small set of pairwise comparisons to train a cost decoder in the latent belief space . A standard model for this is Bradley–Terry:

- A trajectory’s total predicted cost is the sum of per-step costs along imagined or observed belief states.
- The probability that one trajectory is preferred over another increases as its predicted cost decreases.

This makes annotation realistic: the expert does not have to assign numbers, only orderings.

<figure>
  <img src="{{ site.baseurl }}/assets/videos/tinder-preference.gif" alt="Intuitive interface for pairwise trajectory comparisons." style="max-width:350px; display: block; margin: 0 auto;">
  <figcaption style="text-align: center;">
    Experts simply choose which trajectory is safer, making annotation fast and intuitive.
  </figcaption>
</figure>

Now in the low-data regime, we must be careful about uncertainty and underestimation of cost. Here we should be explicitly _pessimistic_.<sup id="footnote6-ref">[[6](#footnote6)]</sup> We use an ensemble of cost decoders and define a pessimistic cost by taking the maximum predicted cost across the ensemble <span id="ref7-ref">[[7](#ref7)]</span>.

If the ensemble disagrees, the pessimistic cost rises. That is exactly what we want: unfamiliar situations are treated as potentially risky.

With preference-based models, we also need a way to set the safety budget. Here, we use the demonstrations again. The method infers the safety budget from the same safe data: if your demonstrations are safe, the inferred budget should be consistent with the costs those demonstrations incur under the learned cost model. This avoids manually guessing a threshold and ties “how safe is safe enough” back to the available data.


<figure>
  <img src="{{ site.baseurl }}/assets/videos/pref_cost_Examples.gif" alt="Examples of trajectory comparisons showing learned cost dynamics." style="width:100%; max-width:900px;">
  <figcaption>
    Visualizing the learned cost dynamics. The violations from the learned cost (red) align well with ground-truth violations (blue).
  </figcaption>
</figure>


Above was a short summary of both constraint learning options. <!-- For more details, check out the [Related Publications](#related-publications) section below. -->  Once we learn the constraints, as already mentioned, we can use them during planning to learn safe policies.


## Discussion and takeaways

Once we put safety first, many downstream design choices become clearer. Here are some reflections on the main decisions we made and why.

### 1) Problem formulation: explicit constraints <sup id="footnote7-ref">[[7](#footnote7)]</sup>

Framing surgical decision making as CMDPs is, in my view, strictly better than the most common alternatives in prior work (implicit penalties, reward shaping, or "safe" heuristics baked into the reward). The reason is control and clarity: a CMDP forces you to separate task performance from constraint satisfaction, and it let us tune this tradeoff explicitly via budgets rather than implicitly via reward weights.<sup id="footnote8-ref">[[8](#footnote8)]</sup>

Unfortunately at the time of writing, there are still very few instances of works that abstract these surgical tasks as CMDPs. Most prior works either ignore safety entirely or conflate it with reward by penalizing unsafe events.<sup id="footnote9-ref">[[9](#footnote9)]</sup> Mixing safety constraint and task reward is problematic because it buries safety under task performance and makes tradeoffs implicit and hard to control. 

That said, I don't think CMDPs are enough. They enforce constraints only in expectation. In safety-critical robotics, expectations may be too weak, because rare-but-catastrophic events can remain acceptable in expectation while still being unacceptable clinically. This is why, long-term, one should consider stricter safety formulations such as Chance Constraints, Control Barrier Functions or similar approaches. 

CMDPs are still a good starting point: they are tractable and make the “safety vs. performance” tradeoff explicit. One could see them as a way of steering agents to safe behavior, with the understanding that stricter safety guarantees may be layered on top.


### 2) Learning constraints is often more practical than learning rewards

Learning safety constraints from demonstrations is often more practical than learning reward functions. Constraints transfer better across tasks and are less sensitive to stylistic variation. The heterogeneity of demonstrations is less of a problem when learning constraints than when learning rewards, because we only care about the boundary between safe and unsafe behavior, not about explaining all demonstrated behavior with a single reward function, as is usual in inverse reinforcement learning.

Another benefit of learning constraints is that, depending on the approach, it enables us to relax optimality assumptions on the expert demonstrations.<sup id="footnote4-ref2">[[4](#footnote4)]</sup> We do not need to assume that the expert is (near-) optimal with respect to some unknown reward as required for most inverse reinforcement learning methods; we only need to assume that expert data covers safe boundaries well enough to learn constraints. This is a more realistic assumption in practice. This was one of my motivations for exploring constraint learning in the first place as I had suffered enough trying to make reward IRL work in practice.

Another reason why one might prefer learning constraints over rewards is _trust_. Assume that we are given a ground truth reward function that perfectly captures the task. There is high chance the learned policy will outperform human experts on that reward. But the question that often arises is: will it do so _safely_? If the reward does not capture all safety aspects perfectly, the learned policy may exploit loopholes in the reward to achieve high return at the cost of safety. Clinicians and patients might not trust such policies. But if safety constraints are learned explicitly _from human demonstrations_, they may be more likely to trust that the learned policies will respect safety boundaries, even if they outperform humans on task return. I learned this lesson firsthand when working on some tasks where the agents were able to outperform humans on reward by performing unintuitive maneuvers. They were effective, but not trustworthy to the clinicians. 

I'm not arguing against reward learning in general; just emphasizing to learn constraints from demonstrations when safety is the main concern. It's easier, more robust, and more practical in many cases.

### 3) With explicit safety, model learning becomes hard to avoid 

If constraints must be enforced during planning then some form of model learning is almost inevitable. Some safety formalisms explicitly require a dynamics model; but even within the CMDP framing, a learned dynamics model offers several advantages such as safe exploration, or offline learning from small annotated datasets (as we have shown) where we use unlabeled data to learn the world model.

If you can roll out trajectories in a learned world model, then the number of real interactions needed to evaluate candidate policies can drop dramatically, and in the purely offline case, it can disappear entirely.

Model-based planning also helps avoiding Q-learning scalability issues due to bootstrapping errors, especially in offline settings <span id="ref9-ref">[[9](#ref9)]</span>. Model learning is generally more scalable and although might still suffer from compounding errors, it is often more robust than many alternatives. 

### 4) Planning in latent space is not optional

Planning directly in pixel space is generally impractical, but ignoring image observations is also unrealistic in surgery. Many prior works avoid the issue by assuming access to low dimensional state (poses, forces, handcrafted features). That is convenient for algorithm development, but it does not reflect how clinical systems are instrumented: endoscopic RGB, ultrasound, fluoroscopy, etc., are central sensing modalities and are not going away.

So the practical position is: learn the right latent space, and do everything there. This is not just about compute efficiency; it is about turning a partially observed, high-dimensional control problem into something that can be optimized robustly. Once we map observations to a latent, downstream learning (constraint learning, planning, uncertainty estimation) become easier.

Empirically, RSSM world models have been particularly effective for this setting: the recurrent belief state can capture the history dependence needed for partial observability, and the latent can encode safety-relevant structure even when trained only with reconstruction/dynamics losses (i.e., without explicit cost signals), at least for the classes of constraints we cared about. This does not mean RSSMs are optimal; there may be better inductive biases or representation objectives (maybe forward-backward representations <span id="ref10-ref">[[10](#ref10)]</span>?) for constraint-relevant latents, and there is room for improvement in how these representations are learned.

### 5) Polytope fences and preference costs are complementary, and the right choice depends on labels and scale

The two constraint learning options above are not mutually exclusive, and they are not exhaustive. Which one is appropriate depends on what supervision is available and what failure modes you care about. I tend to choose the preference-based approach as a default, as it seems less sensitive to hyperparameters and generalizes better in practice, but both have their merits.

- Convex polytope safe sets can be attractive when you can label safe vs. unsafe segments. One advantage is interpretability: the constraint is a geometric region, and violations correspond to leaving that region in latent space. In addition, at least theoretically, this approach should learn multiple unknown constraints, but in practice we have not fully tested this multi-constraint interpretation.

  One current drawback is scalability. Increasing the number of facets increases representational power, but it also increases optimization complexity at planning time. In our constrained planning setup, more facets effectively means more constraint terms and, depending on the solver, more Lagrange multipliers or more complex constrained updates. At the time of writing, it is unclear how to optimize with large numbers of facets (or learned costs) during planning without introducing significant complications such as competing objectives and instabilities.
  
  One limitation of the CPM-based approach is that it requires both safe _and_ unsafe examples to learn the polytope boundary. One could design alternative polytope construction algorithms that work from safe examples alone (without being too conservative), or augment the dataset by generating artificial unsafe examples from simulations. This would be particularly attractive in clinical settings where collecting real unsafe demonstrations is undesirable.

  One reason I'm less enthusiastic about polytope methods as a default is that we lose all theoretical guarantees with the approximations we make for tractability. In practice, the learned polytopes work well, but then one could argue that preference-based costs work well too, and they are more flexible. If someone shows that polytope methods do indeed capture more than one constraint reliably, that would make them more compelling.


- Preference-based constraint learning can be the better default when unsafe examples are unavailable. Pairwise comparisons tend to be more natural for clinicians than absolute scoring, and preference learning has also proven effective at scale in other domains.

  The main concern is coverage: learning a safety boundary from preferences over trajectories generated by only safe policies can underestimate cost in parts of the state space never visited. In practice, large and diverse datasets can mitigate this, but it remains a conceptual weakness if the dataset fails to cover near-boundary behavior. Another subtle issue is conflict: if preferences implicitly mix multiple unknown costs that sometimes disagree (e.g., “minimize tissue stress” vs. “minimize procedure time”), the learned cost may become context-dependent or inconsistent unless the model and labeling protocol explicitly represent that structure.

  It would be valuable to test these ideas with real preference data from experts, including disagreement analysis across surgeons and across clinical contexts. That would reveal whether the learned constraints reflect stable “guardrails” or whether safety is inherently multi-objective and context-conditioned in a way we must model explicitly.

Maybe repeating myself in a shorter form:
### Practical takeaways

- Treat safety as an explicit constraint first; it clarifies what must be learned and what must be enforced.
- CMDPs are a natural way to frame safety constraints. Prefer them over vanilla MDPs with scalarized rewards. Safety in expectation constraints can be too weak however; chance constraints/CBFs could be natural next steps.
- Learning safety constraints from demonstrations is often more practical than learning reward functions. Constraints transfer better across tasks and are less sensitive to stylistic variation. They also require more realistic assumptions on the data.
- World models offer many advantages for safe learning. Purely model-free, on-policy methods (e.g., vanilla PPO) are often a poor fit for high-stakes domains unless paired with strong safety mechanisms.
- High-dimensional data is unavoidable in surgery; the right abstraction is to plan and enforce safety in learned latent belief space.
- Polytope constraints and preference costs trade off interpretability, supervision requirements, and scalability; choose based on what labels and compute you can afford. In any case, pessimism is crucial for safety.
- **Consider seriously the suggested pipeline**: offline feature learning, offline constraint learning, then constrained model-based planning. It minimizes real interactions and annotation while maximizing safety. 

### Future work
From here there are many directions to explore. Some that I find particularly interesting:
- Stricter safety guarantees: CMDPs are a start, but not enough for high-stakes settings. Chance constraints, CVaR, or Control Barrier Functions could be next steps.
- Better latent representations: RSSMs work well, but are they optimal for all types of safety signals we care about? Are there better inductive biases or objectives for learning safety-relevant latents?
- Alternative polytope learning methods: can we learn conservative polytopes from safe data alone? Can we scale to many facets without complicating planning?
- Multi-constraint learning: real surgical procedures have many safety constraints that may conflict. Learning multiple constraints explicitly, and reasoning about tradeoffs between them, is an important next step.
- Real expert data: testing these methods with real clinical demonstrations and preference labels would validate their practical utility and reveal real-world challenges.
- Real deployment: integrating learned constraints into real surgical robotic systems and evaluating their impact on safety and performance in clinical settings is the ultimate goal. One could start by only providing safety suggestions and context to human operators rather than full autonomy.

If you are interested in any of these directions, feel free to reach out!

---

## Footnotes

<div style="font-size: 0.9em; color: #555;">

<p><a id="footnote1"></a>[1] Of course, the same question applies to inferring reward functions from data, but that is outside the scope of this post. <a href="#footnote1-ref">↩</a></p>

<p><a id="footnote2"></a>[2] Throughout this post, I use the term <em>safe</em> both in the formal CMDP sense (expected cost below budget) and in the more general sense of "not causing harm" or "respecting clinical safety constraints". It might be imprecise or confusing to conflate these meanings, but I hope the context makes it clear which sense is intended. <a href="#footnote2-ref">↩</a></p>

<p><a id="footnote3"></a>[3] Actually, to be precise, we know that convex hull of feature expectation of safe policies define a safe set, under some assumptions, such as linearity of costs in features, correct estimation, etc. See Lindner et al. <a href="#ref4">[4]</a> for details. <a href="#footnote3-ref">↩</a></p>

<p><a id="footnote4"></a>[4] Safety certification does not rely on reward-optimal demonstrations, but obtaining guarantees about downstream reward performance typically does require additional assumptions (coverage and, in some analyses, noisy/near-optimality). <a href="#footnote4-ref1">↩</a> <a href="#footnote4-ref2">↩<sup>2</sup></a></p>

<p><a id="footnote5"></a>[5] There are also other reasons for losing these guarantees, for example our learned features might not be linear in the unknown costs. <a href="#footnote5-ref">↩</a></p>

<p><a id="footnote6"></a>[6] Note that in the polytope-based method, pessimism is built in by design. <a href="#footnote6-ref">↩</a></p>

<p><a id="footnote7"></a>[7] One might argue that "reward is enough" <span id="ref8-ref"><a href="#ref8">[8]</a></span> and that any safety constraint can be encoded in a sufficiently well-designed reward function. While this may hold theoretically under <em>certain</em> assumptions, it is beside the point for our purposes. In high-stakes settings, interpretability, explicit control over safety tradeoffs, and the ability to audit constraint satisfaction separately from task performance are important requirements. <a href="#footnote7-ref">↩</a></p>

<p><a id="footnote8"></a>[8] Of course, in practice, choosing the right budget value can be non-trivial. Without units or interpretable semantics attached to the cost function, a budget threshold is just a number, and tuning it still requires domain insight or iterative calibration against safe behavior. <a href="#footnote8-ref">↩</a></p>

<p><a id="footnote9"></a>[9] There are works that incorporate explicit constraints or verification layers, but they are not yet widespread. <a href="#footnote9-ref">↩</a></p>

</div>

---
## References

<div style="font-size: 0.9em; color: #555;">

<p><a id="ref1"></a>[1] Altman, E. (1999). Constrained Markov Decision Processes. <em>CRC Press</em>. [<a href="https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780849303821">Book</a>] <a href="#ref1-ref">↩</a></p>

<p><a id="ref2"></a>[2] Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and Acting in Partially Observable Stochastic Domains. <em>Artificial Intelligence</em>, 101(1-2), 99-134. [<a href="https://doi.org/10.1016/S0004-3702(98)00023-X">Paper</a>] <a href="#ref2-ref">↩</a></p>

<p><a id="ref3"></a>[3] Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. <em>International Conference on Learning Representations (ICLR)</em>. [<a href="https://arxiv.org/abs/1912.01603">Paper</a>] <a href="#ref3-ref">↩</a></p>

<p><a id="ref4"></a>[4] Lindner, D., Turchetta, M., Tschiatschek, S., Berkenkamp, F., & Krause, A. (2024). Learning Safety Constraints from Demonstrations with Unknown Rewards. <em>Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 238:1378-1386</em>. [<a href="https://proceedings.mlr.press/v238/lindner24a.html">Paper</a>] [<a href="https://arxiv.org/abs/2011.05552">arXiv</a>] <a href="#ref4-ref">↩</a></p>

<p><a id="ref5"></a>[5] Kantchelian, A., Tschantz, M. C., Huang, L., Bartlett, P. L., Joseph, A. D., & Tygar, J. D. (2014). Large-Margin Convex Polytope Machine. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>. [<a href="https://proceedings.neurips.cc/paper/2014/hash/e0cf1f47118daebc5b16269099ad7347-Abstract.html">Paper</a>] <a href="#ref5-ref">↩</a></p>

<p><a id="ref6"></a>[6] Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. <em>Biometrika</em>, 39(3/4), 324-345. [<a href="https://www.jstor.org/stable/2334029">Paper</a>] <a href="#ref6-ref">↩</a></p>

<p><a id="ref7"></a>[7] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>. [<a href="https://arxiv.org/abs/1612.01474">Paper</a>] <a href="#ref7-ref">↩</a></p>

<p><a id="ref8"></a>[8] Silver, D., Singh, S., Precup, D., & Sutton, R. S. (2021). Reward is enough. <em>Artificial Intelligence</em>, 299, 103535. [<a href="https://doi.org/10.1016/j.artint.2021.103535">Paper</a>] <a href="#ref8-ref">↩</a></p>

<p><a id="ref9"></a>[9] Park, S., Frans, K., Mann, D., Eysenbach, B., Kumar, A., & Levine, S. (2025). Horizon Reduction Makes RL Scalable. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>. [<a href="https://arxiv.org/abs/2406.04168">Paper</a>] <a href="#ref9-ref">↩</a></p>

<p><a id="ref10"></a>[10] Touati, A., & Ollivier, Y. (2021). Learning One Representation to Optimize All Rewards. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>. [<a href="https://arxiv.org/abs/2103.07945">Paper</a>] <a href="#ref10-ref">↩</a></p>

</div>

---


## Related Publications
The arxiv versions will be added here as soon as they are online.

<!--
<div class="citation-box">
<p><strong>Polytope-based Safety Constraints</strong></p>
<p style="font-size: 0.9em; color: #555; margin-bottom: 0.5rem;">[Author Name]. ([Year]). [Title of Polytope Manuscript]. <em>[Venue/Journal]</em>.</p>
<pre><code>@article{authorname2025polytope,
  title={[Title of Polytope Manuscript]},
  author={[Author Name] and [Co-authors]},
  journal={[Venue/Journal]},
  year={[Year]}
}</code></pre>
</div>

<div class="citation-box">
<p><strong>Preference-based Constraint Learning</strong></p>
<p style="font-size: 0.9em; color: #555; margin-bottom: 0.5rem;">[Author Name]. ([Year]). [Title of Preference Learning Manuscript]. <em>[Venue/Journal]</em>.</p>
<pre><code>@article{authorname2025preference,
  title={[Title of Preference Learning Manuscript]},
  author={[Author Name] and [Co-authors]},
  journal={[Venue/Journal]},
  year={[Year]}
}</code></pre>
-->
