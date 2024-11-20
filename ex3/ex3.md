# 4.2 Poincare maps

When you have an n-dimensional system $x = f(x)$, a Surface of Section $S$ is an
$(n-1)$--dimensional subspace, chosen such that for a trajectory of interest,
there are $\{t_i\}: t \in \{t_i\} \iff x(t_i) \in S$. Also, $S$ is not tangent 
to the trajectory. A Poincare map $P$ for $S$ and the dynamical system $x=f(x)$ 
is then defined via $P(x(t_i)) = x(t_i+1)$.

## 4.2.1

Let $\dot x = f(x)$ be a dynamical system and $P$ a corresponding Poincare map
with surface of section $S$.

$$
\exists \ y \in S: P(y) = y
\implies
\text{there exists a closed cycle in the system}
$$

Is that also a necessary condition?

---

On a closed cycle, $x=x(t)$ is a $T$-periodic function, where $T=\min \{T'\}$.

Let $(x,t_0)$ be a certain point of intersection between $S$ and our trajectory.
Point $(x,t_0+T)$ is also an intersection, thus $(t_0 + T) \in \{t_i\}, i>0$.
If $t_0 + T = t_1$, then $P(x(t_0)) = x(t_0 + T) = x(t_0)$.
Otherwise, the condition does not follow - only the more general form for 
n-cycles with $P^n$.

## 4.2.2

$$
\dot r = r (1-r), \quad \dot \theta = 2
$$

$$
S = \{(r,\theta): r>0, \theta=0\}
$$

Prove that there exists a closed cycle at $r = 1$. Describe stability.

---

$$
\begin{gathered}
    T = \frac{2\pi}{\dot \theta} \\
    \frac{dr}{r(1-r)} = dt \\
    \ln \left| \frac{r}{1-r} \right| = t + C \\
    P(r_i) = \frac{r_i e^\pi}{1 + r_i (e^\pi - 1)} \\
\end{gathered}
$$

Plugging $r=1$ in, we get $P(1) = 1$. 

$$
\frac{e^\pi}{((e^\pi-1)r+1)^2}
$$

Since $|P'(1)| = |e{-\pi}| < 1$, the fixed point is stable.
