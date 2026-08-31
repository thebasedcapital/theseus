# Theseus V0 - mathematical formulation

## 1. State is a checkpoint artifact, not only a function

Let a model checkpoint be a state

\[
s=(\theta, a) \in \mathcal X,
\]

where \(\theta\) is the parameter vector and \(a\) can include artifact-level state such as quantization scales, optimizer state, adapter structure, or other metadata.

This distinction matters because two parameterizations can implement the same predictor while reacting differently to later operations.

For a one-hidden-layer ReLU network

\[
f_\theta(x)=W_2\,\sigma(W_1x+b_1)+b_2,
\]

positive homogeneity gives

\[
\sigma(Dz)=D\sigma(z) \qquad D=\operatorname{diag}(d_i),\ d_i>0.
\]

Therefore the transformation

\[
W_1' = DW_1,\qquad b_1'=Db_1,\qquad W_2'=W_2D^{-1}
\]

leaves the realized function unchanged:

\[
f_{\theta'}(x)=f_\theta(x).
\]

The prototype deliberately moves along this gauge orbit. The current predictor stays fixed while the future behavior of concrete surgery operators changes.

This is consistent with recent work on positive-homogeneous gauge redundancy, path-conditioned training, and hidden-gauge effects on training dynamics. It is also exactly why a lifecycle benchmark must specify the **operator** acting on the artifact rather than only the function represented by the artifact.

---

## 2. Each model surgery is a controlled dynamical system

For operation family \(o\), write

\[
s_{t+1}=F_o(s_t,u_t,\xi_t),
\]

where

- \(u_t\) is a controllable setting such as learning rate, number of steps, pruning ratio, merge coefficient, bit-width, adapter rank, or regularization strength;
- \(\xi_t\) captures stochasticity such as minibatch sampling or calibration data.

Examples:

\[
o\in\{\text{SFT},\text{DPO},\text{merge},\text{prune},\text{quantize},\text{unlearn},\text{edit},\text{distill}\}.
\]

Define a **safe set** of checkpoints whose protected capabilities remain acceptable:

\[
\mathcal K
=
\{s: U_j(s)\ge c_j\ \forall j\}.
\]

For each operation, define a target set

\[
\mathcal T_o
=
\{s: G_o(s)\ge \tau_o\},
\]

where \(G_o\) is the desired effect of that surgery.

Examples:

- SFT target: new-domain accuracy above a threshold.
- Unlearning target: forget-set score below a threshold.
- Quantization target: target bit-width reached.
- Merge target: both parent capabilities retained.

---

## 3. Operation-specific capture basin

Viability theory gives the right object.

For finite horizon \(H\), define the capture basin

\[
\mathcal C_o^H
=
\left\{
 s_0\in\mathcal K:
 \exists u_{0:H-1}\text{ such that }
 s_t\in\mathcal K\ \forall t<H,
 \ s_H\in\mathcal T_o
\right\}.
\]

Interpretation:

> \(s\in\mathcal C_o^H\) means this exact checkpoint can still undergo operation \(o\), under the allowed control budget, without violating protected capabilities.

This is the direct analogue of a constrained control system that can still reach its goal without leaving its safe region.

The binary optionality vector is

\[
\mathbf z(s)
=
[
\mathbf 1(s\in\mathcal C_1^H),\ldots,
\mathbf 1(s\in\mathcal C_m^H)
].
\]

A first scalar summary is simply

\[
\Omega_0(s)
=
\frac1m\sum_{o=1}^m \mathbf 1(s\in\mathcal C_o^H).
\]

The V0 smoke test uses this deliberately crude version because it is hard to fake and easy to interpret.

---

## 4. Replace binary membership with a quantitative reserve

Binary controllability is usually too coarse. Control theory replaces yes/no controllability with minimum-energy reachability. We can do the same.

Let \(c_o(u_t)\) be the cost of a surgery control. Define minimum operation cost

\[
J_o^*(s)
=
\inf_{u_{0:H-1}}
\sum_{t=0}^{H-1} c_o(u_t)
\]

subject to

\[
s_t\in\mathcal K,\qquad s_H\in\mathcal T_o.
\]

Given budget \(B_o\), define an operation reserve

\[
R_o(s;B_o)
=
\begin{cases}
\max\left(0,1-\frac{J_o^*(s)}{B_o}\right), & J_o^*(s)<\infty,\\
0,& \text{otherwise}.
\end{cases}
\]

Then the model optionality vector is

\[
\mathbf R(s)
=
[R_1(s),\ldots,R_m(s)].
\]

A weighted scalar option value is

\[
\Omega(s)
=
\sum_o p_oR_o(s),
\]

where \(p_o\) is the expected future importance of each operation.

This makes the score deployment-specific without changing the underlying benchmark data.

---

## 5. Robust reserve and distance to losing an operation

Control theory also uses a **distance to uncontrollability**. The analogous quantity here is

\[
\rho_o(s)
=
\inf_{s'\notin\mathcal C_o}
 d(s,s').
\]

A large \(\rho_o\) means operation \(o\) remains feasible under perturbation.

However, ordinary Euclidean distance in parameter space is not a valid default because ReLU gauge transformations can move arbitrarily far in parameter space without changing the represented function.

Candidate symmetry-aware distances include:

- path metrics for positively homogeneous networks;
- output/logit distribution distances on a calibration set;
- Fisher or natural-gradient metrics;
- architecture-specific canonical coordinates.

The prototype exposes this issue rather than hiding it.

---

## 6. Artifact optionality vs intrinsic optionality

This is the strongest mathematical lesson from V0.

Let \([\theta]\) denote the equivalence class of parameters implementing the same function under a known symmetry group \(\mathcal G\).

Concrete operators such as magnitude pruning, per-tensor quantization, ordinary SGD, and raw weight averaging are generally **not** invariant under \(\mathcal G\).

So define two quantities.

### Artifact optionality

\[
R_o^{\text{artifact}}(\theta)
\]

measures readiness of the checkpoint exactly as stored.

### Canonical optionality

Choose a deterministic gauge-fixing map \(C\) that selects a representative of each orbit:

\[
C([\theta])=\theta_{\text{canon}}.
\]

Then

\[
R_o^{\text{canon}}([\theta])
=
R_o^{\text{artifact}}(C([\theta])).
\]

The difference

\[
D_o^{\text{gauge}}(\theta)
=
R_o^{\text{canon}}([\theta])
-
R_o^{\text{artifact}}(\theta)
\]

is **avoidable lifecycle debt** caused by representation rather than function.

That is exactly what the smoke test demonstrates.

---

## 7. Closed-form gauge fixing used in V0

For hidden unit \(i\), let

\[
a_i=\sqrt{\|W_{1,i:}\|_2^2+b_{1,i}^2},
\qquad
b_i=\|W_{2,:,i}\|_2.
\]

Under rescaling by \(s_i>0\):

\[
a_i' = s_i a_i,
\qquad
b_i' = b_i/s_i.
\]

Balancing them gives

\[
s_i^* = \sqrt{b_i/a_i},
\]

and therefore

\[
a_i'=b_i'=\sqrt{a_ib_i}.
\]

This is a function-preserving canonicalization step. In the smoke test it converts a checkpoint that passes 0/4 future-operation tests back into one that passes 4/4, without changing current predictions.

---

## 8. Local differentiable readiness

For gradient-based operations, recent work defines Optimization Readiness (OR) using gradient strength and gradient reliability:

\[
S(\theta;\mathcal T)
=
\frac{\|g\|_2^2}{L},
\]

\[
Q(\theta;\mathcal T)
=
\frac{\|g\|_2^2}{\mathbb E_B\|\hat g_B\|_2^2},
\]

\[
\operatorname{OR}=SQ.
\]

Under smoothness assumptions it lower-bounds one-step relative optimization gain for an appropriate step size.

For Theseus, the natural extension is **operation-conditioned readiness**. If a differentiable surgery can only move in subspace \(\mathcal S_o\) with orthogonal projector \(P_o\), define

\[
S_o
=
\frac{\|P_og\|_2^2}{L},
\qquad
Q_o
=
\frac{\|P_og\|_2^2}
{\mathbb E_B\|P_o\hat g_B\|_2^2},
\]

\[
\operatorname{OR}_o=S_oQ_o.
\]

Examples of \(P_o\):

- adapter subspace for LoRA;
- null space protecting retained knowledge;
- editable parameter mask;
- selected layers in partial fine-tuning.

For nonlinear/discrete operations such as quantization and pruning, empirical capture tests remain the safer V0 definition.

---

## 9. The prototype's falsifiable hypothesis

The smoke test checks the strongest possible version of the idea:

\[
f_{\theta_A}\approx f_{\theta_B}
\]

on all current inputs, while

\[
\mathbf R(\theta_A)\ne\mathbf R(\theta_B).
\]

In fact, \(\theta_B\) is generated from \(\theta_A\) by an exact symmetry, so the difference in future operation reserve cannot be blamed on current task quality.

The observed result is:

\[
\Omega_0(\theta_A)=1,
\qquad
\Omega_0(\theta_B)=0,
\qquad
\Omega_0(C(\theta_B))=1.
\]

That establishes a minimal form of hidden lifecycle state.

---

## 10. Prior work that constrains our novelty claims

The following ideas are prior art and should be treated as ingredients, not claimed as new:

- Viability kernels / capture basins and constrained reachability: Aubin, Frankowska, Saint-Pierre and the control-theory literature.
- Quantitative controllability via minimum energy and controllability-Gramian spectra.
- Optimization Readiness for future trainability: arXiv:2605.09044.
- Quantization designed for downstream adapter correctability: ProjQ, arXiv:2606.00494.
- Positive-homogeneous gauge redundancy and gauge fixing: arXiv:2602.14729 and related path-space work.
- Functionally identical gauges producing different training dynamics: arXiv:2608.06766.
- Rescaling-invariant path metrics for pruning / quantization analysis: PMLR 2025 work by Gonon et al.
- Sequential bounded plasticity decay: CellFill / In-Cell Learning, arXiv:2608.20873.

The candidate new object is the **heterogeneous operation-specific lifecycle surface** across learning, alignment, merging, editing, unlearning, pruning, quantization, and distillation, with both artifact-level and symmetry-canonical views.
