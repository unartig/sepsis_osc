#import "../thesis_env.typ": *
#import "../figures/kuramoto.typ": kuramoto_fig

= Model Background \ (Dynamic Network Model) <sec:dnm>

Macroscopic multi-organ failure associated with sepsis is driven by a dysregulated cascade of signaling processes on a microscopic level (see @sec:sepbio).
This cascade involves a massive amount of interconnected components, where the connections mechanics and strengths vary over time and space.
For example, these interactions differ across tissues and evolve as sepsis progresses, with crossing biochemical thresholds the behavior of cells can be changed @Callard1999Cytokines.

In essence, cell-to-cell and cell-to-organ interactions in septic conditions form a highly dynamic, nonlinear and spatio-temporal network of relationships @Schuurman2023Complex, which cannot be fully understood by a reduction to single time-point analyzes.
Even though many individual elements of the inflammatory response are well characterized, we still fail to integrate them into a coherent system-level picture.

To address this complexity, the emerging field of _Network Physiology_ provides a promising conceptual framework.
Rather than studying components in isolation, network physiology focuses on the coordination and interconnection among the diverse organ systems and subsystems @Ivanov2021Physiolome.
It enables the study of human physiology as a complex, integrated system, where emergent macroscopic dynamics arise from interacting subsystems that cannot be explained by their individual behavior.
This perspective translates to the mesoscopic level, i.e. the in-between of things, where the coupling mechanisms collectively determine the overall physiological function.

In network physiology, the analytical basis of the bodies interacting systems is often graph based.
Nodes represent subsystem such as organs or cell populations and links represent functional couplings or communication pathways @Ivanov2021Physiolome.
Unlike classical graph theory, where dynamics are introduced by changing the graph topology (e.g. adding or removing links or nodes), in _Complex Networks_ the links themselves can evolve dynamically in response to other system variables.
These adaptive connections allow for information to propagate through the whole network, giving rise to emerging phenomena on global scales for otherwise identical network topologies.

Complex networks are well studied in physics and biology and have been applied to various physiological domains.
Early works studied the cardiovascular system @Guyton1972Circulation, while more recent studies have focused on the cardio-respiratory coupling @Bartsch2012Phase and large-scale brain network dynamics @Lehnertz2021Time.
Network approaches have also provided mechanistic insights into disease dynamics, for example Parkinson @Asl2022Parkinson and Epilepsy @Simha2022Epilepsy.

Building on these interaction-centric principles has opened up new opportunities to study how the inflammatory processes, such as those underlying sepsis, emerge from the complex inter- and intra-organ communication.
In particular Sawicki et. al @Sawicki2022DNM and Berner et. al @Berner2022Critical have introduced a dynamical system that aims to model the cytokine behavior in patients with sepsis and cancer.
This functional model will be referred to as #acl("DNM") and forms the conceptual foundation for this whole project.

The remainder of this chapter is structured as follows: @sec:kuramoto introduces the theoretical backbone of the #acr("DNM"), the Kuramoto oscillator model, which provides a minimal description of synchronization phenomena in complex systems.
@sec:dnmdesc presents the formal mathematical definition of the #acr("DNM") and its medical interpretation, followed by implementation details in @sec:dnmimp and a presentation of selected simulation results in @sec:dnmres.

== Theoretical Background: The Kuramoto Oscillator Model <sec:kuramoto>
To mathematically describe natural or technological phenomena, _coupled oscillators_ have proven to be a useful framework, for example, to model the relative timing of neural spiking, reaction rates of chemical systems or dynamics of epidemics @pikovsky2001synchronization.
In these cases, complex networks of coupled oscillators are often capable of bridging microscopic dynamics and macroscopic synchronization phenomena observed in biological systems.

One of the most influential system of coupled oscillators is the _Kuramoto Phase Oscillator Model_.
This system of coupled #acr("ODE"), where each term describes an oscillator rotating around the unit circle, is often used to study how synchronization emerges from simple coupling rules.
In the simplest form, it consists of $N in NN_(>0)$ identical, fully connected and coupled oscillators with phase $phi_i in [0, 2pi)$, for $i = 1,...,N$ and an intrinsic frequency $omega_i in RR$ @kuramoto1984chemical.
The phase $phi_i$ represents the angular position on the unit circle, hence it is defined $"modulo "2pi$.
Traditionally, two oscillators $i$ and $j$ are coupled through the sine function $sin(phi_i - phi_j)$, the dynamics are then given by the #acr("ODE") system:
$
  dot(phi)_i = omega_i - Kappa/N sum^N_(j=1) sin(phi_i - phi_j)
$ <eq:kuramoto>

Here, the $dot(phi)$ is used as shorthand notation for the time derivative of the phase $(d phi)/(d t)$, the instantaneous phase velocity.
An additional parameter is the global coupling strength #box($Kappa in RR_(>0)$) between oscillators $i$ and $j in 1...N$.
The system evolution depends on the choice of initial phases $phi_i (t=0)$, which are typically drawn from a uniform random distribution over $[0, 2pi)$.

When evolving this system with time, commonly by numerical integration, since analytical solutions only exist for special cases, oscillator $i$'s phase velocity depends on each other oscillator $j$.
The sine coupling $sin(phi_i - phi_j)$ implements a phase-cohesion mechanism: oscillator $i$ decelerates when it leads oscillator $j$ ($0 < phi_i - phi_j < pi$) and accelerates when it lags behind ($-pi < phi_i - phi_j < 0$), pulling the population towards synchronization. 

For sufficiently large $N$, the oscillator population can converge towards system-scale states of coherence or incoherence based on the choice of $Kappa$ @acebron2005kuramoto.
Coherent in this case means oscillators synchronize with each other, so they share the same phase and phase velocity, incoherence on the other hand is the absence of synchronization (desynchronized), see @fig:sync.
Synchronous states can be reached if the coupling is stronger than a certain threshold $Kappa>Kappa_c$, the critical coupling strength @acebron2005kuramoto.
In between these two regimes there is a transition-phase of partial synchronization, where some oscillators phase- and frequency-lock and others do not.
The model captures the mechanism of self-synchronization, and a collective transition from disorder to order, that underlies many real world processes, which is the reason the model has attracted so much research @pikovsky2001synchronization.

#figure(
  scale(kuramoto_fig, 110%),
  caption: flex-caption(
  short: [Synchronization transition in the Kuramoto model],
  long: [Illustration of the collective dynamics of a population of phase oscillators on the unit circle as the global coupling strength $Kappa$ is increased.
         Each point represents an oscillator at phase $phi_i$, color encodes the instantaneous phase velocity $dot(phi)_i$ (blue/yellow: faster, green: slower).
         For weak coupling (left), oscillators are desynchronized, with phases distributed around the circle and heterogeneous frequencies.
         At intermediate coupling (center), partial synchronization emerges: a subset of oscillators forms a coherent cluster that phase- and frequency-locks, while the remaining oscillators drift incoherently.
         For sufficiently strong coupling (right), the population becomes fully synchronized, with all oscillators sharing a common phase and frequency.
  ]),
) <fig:sync>


=== Extensions to the Kuramoto Model <sec:extent>
To increase the generalization ability of the system, various extensions of the basic Kuramoto model have been proposed and studied both numerically and analytically.
Several extensions are directly relevant to the #acr("DNM") and their definitions and effects on synchronization will be introduced, with additional terms being indicated by the red color:

*Phase Lag $alpha$*, introduced in @sakaguchi1986rotator, where $abs(alpha) < pi/2$.
By shifting the coupling function, values of $alpha != 0$ act as an inhibitor of synchronization, so the coupling does not vanish even when the phases align:
$
  dot(phi)_i = omega_i - Kappa/N sum^N_(j=1) sin(phi_i - phi_j cmred(+ alpha))
$
As a result the critical coupling strength $K_c$ increases with $alpha$ @sakaguchi1986rotator.

*Adaptive coupling $bold(Kappa) in [-1, 1]^(N times N)$* moves from a global coupling strength $Kappa$ for all oscillator pairs to an adaptive asymmetric coupling strength for each individual pair #box($bold(Kappa) = (kappa_(i j)) _((i,j)in N times N)$):
$
  dot(phi)_i = omega_i - 1/N sum^N_(j=1) cmred(kappa_(i j)) sin(phi_i - phi_j) \
  cmred(dot(kappa)_(i j) = - epsilon (kappa_(i j) + sin(phi_i - phi_j + beta^mu)))
$ <eq:kurasaka>
Here, the adaptation rate $0 < epsilon << 1$ separates the fast moving oscillator dynamics from slower moving coupling adaptivity @Berner2020Birth.
Such adaptive couplings have been used to model neural plasticity and learning-like processes in physiological systems @Jttner2023Dynamic.
The so called phase lag parameter $beta$ of the adaptation function (also called plasticity rule) plays an essential role in the synchronization process.
At a value of $beta^mu=pi/2$, the coupling, and therefore the adaptivity, is at a maximum positive feedback, strengthening the link $kappa_(i j)$ and encouraging synchronization between oscillators $i$ and $j$.
This maximal connectivity is referred to _Hebbian Rule_ and found in synchronizing systems such as the brain  @Berner2020Birth.
For other values $beta^mu != pi/2$, the feedback is delayed $phi^(mu)_i-phi^(nu)_j=beta^mu-pi/2$ by a phase lag, at a value of $beta^mu=-pi/2$ we get an anti-Hebbian rule which inhibits synchronization.

*Multiplex Networks* represent systems with multiple interacting layers.
Multiplexing introduces a way to couple $L in NN_(>1)$ Kuramoto networks (indexed by $mu=1,...,L$ and $nu=1,...,L!=mu$) via interlayer links:
$
  dot(phi)_i^cmred(mu) = omega_i - Kappa/N sum^N_(j=1) sin(phi_i - phi_j cmred(+ alpha^(mu))) cmred(- sigma^(mu nu) sum^L_(nu=1, nu!=mu) sin(phi_i^mu - phi_i^nu + alpha^(mu nu)))
$
Here $mu$ and $nu$ represent distinct subsystems, and are connected via interlayer coupling weights $sigma^(mu nu) in RR_(>=0)$, where oscillators are acting one-to-one.\

These extensions combined serve as the source of dynamics for the #acr("DNM") and give rise to more intricate system states than the straightforward synchronization in the base model.
Even for single layers, non-multiplexed but phase-lagged and adaptively coupled oscillators, one can observe several distinct system states, neither fully synchronized or desynchronized such as phase and frequency-clusters, chimera- and splay states.
The emergence of these states depends on the choice of the coupling strength $Kappa$ and the phase-lag parameters $alpha$ and $beta$ and sample distribution of initial phases.

In the frequency clustered state, the oscillator phases do not synchronize, but several oscillator groups can form that share a common frequency.
For the phase-clustered case, the groups additionally synchronize their phase.
Frequency clusters often emerge as intermediate regimes between full synchronization and incoherence @Berner2019Hiera.

Chimera states, a special type of partial synchronization, occur when only a subset of oscillators synchronizes in phase and frequency, while others remain desynchronized.
In contrast to "normal" partial synchronization they occur when the coupling symmetry breaks.
In splay states, all oscillators synchronize their frequencies but do not their phases, they instead uniformly distribute around the unit circle @Berner2020Birth.

The introduction of multiplexed layers changes the system behavior once more, for example individual layers of a multiplexed system can result in the multi-clustered regime for parameters they would not in the monoplexed case.
In multiplexed systems it is also possible connected layers end up in different stable states, for example, one in a clustered the other in a splay state.

== Description <sec:dnmdesc>
Built on top of the Kuramoto Oscillator Model, the #acr("DNM") is a *functional* model, that means it *does not try to model things accurately on any cellular, biochemical, or organ level*, it instead tries to model dynamic interactions.
At the core, the model does differentiate between two broad classes of cells, introduced in @sec:cell, the stroma and the parenchymal cells.
It also includes the cell interaction through cytokine proteins and an information flow through the basal membrane.
Importantly, the model only handles the case of already infected subjects and tries to grasp if the patients state is prone to a dysregulated host response.

Cells of one type are aggregated into layers, everything associated with parenchymal cells is indicated with an $dot^1$ superscript and is called the _organ layer_, stroma cells are indicated with $dot^2$ and is referred to as nonspecific _immune layer_.
Each layer consists of $N$ phase oscillators $phi^ot_i in [0, 2pi)$.
To emphasize again the function aspect of the model: individual oscillators do not correspond to single cells, rather the layer as a whole is associated with the overall state of all organs or immune system functionality respectively.
An illustration of the biological and functional model is shown in @fig:dnm.

#figure(
  scale(image("../images/critical_fig1.png"), 90%),
  caption: flex-caption(
  short: [Schematic illustration of the #acl("DNM")],
  long: [Image taken from @Berner2022Critical
 in a simplified form, illustrating the #acl("DNM").
  *A* A tissue element is depicted, in which the basic processes of sepsis take place.
  Shown are different immune and structural cells involved (colored) in the stroma (yellow area), the parenchym (grey), and the capillary blood vessel, where cytokines, #acr("PAMP")s and #acr("DAMP")s are transmitted.
  *B* Depicts the functional interactions within and between the two corresponding network layers in the #acr("DNM"), the parenchyma and the stroma (immune layer).
  ]),
) <fig:dnm>

The metabolic cell activity is modeled by rotational velocity $dot(phi)$ of the oscillators, the faster the rotation, the faster the metabolism.
Each layer is fully coupled via an adaptive possibly asymmetric matrix $bold(Kappa)^ot in [-1, 1]^(N times N)$ with elements $kappa^ot_(i j)$, these couplings represent the activity of cytokine mediation.
Small absolute coupling values indicate a low communication via cytokines and grows with larger coupling strength.
For the organ layer there is an additional non-adaptive coupling part $bold(A)^1 in [0, 1]^(N times N)$ with elements $a^1_(i j)$, representing a fixed connectivity within an organ.

The dimensionless system dynamics are described with the following coupled #acr("ODE") terms, build on the classical Kuramoto model described in @sec:kuramoto and its extensions from @sec:extent:

// I know this is awful, but please typst, fix your align / sub-numbering
$
  dot(phi)^1_i =& omega^1 - 1/N sum^N_(j=1) lr({ (a^1_(i j) + kappa^1_(i j))sin(phi^1_i - phi^1_j + alpha^(11)) }) - sigma sin(phi^1_i - phi^2_i + alpha^(12)) #<odep1> \
$$
  dot(kappa)^1_(i j) &= -epsilon^1 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) quad quad quad quad quad quad #<odek1> \
$$
  dot(phi)^2_i =& omega^2 - 1/N sum^N_(j=1) lr({kappa^2_(i j)sin(phi^2_i - phi^2_j + alpha^(22)) }) - sigma sin(phi^2_i - phi^1_i + alpha^(21)) #<odep2> \
$$
  dot(kappa)^2_(i j) &= -epsilon^2 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)) quad quad quad quad quad quad #<odek2>
$ <eq:ode-sys>
Where the interlayer coupling, i.e. a symmetric information through the basal lamina, is modeled by the parameter $sigma in RR_(>=0)$.
The internal oscillator frequencies are modeled by the parameters $omega^ot$ and correspond to a natural metabolic activity.

Besides the coupling weights in $bold(Kappa)^ot$ the intralayer interactions also depend on the phase lag parameters $alpha^11$ and $alpha^22$ modeling cellular reaction delay.
To separate the fast moving oscillator dynamics from the slower moving coupling weights adaptation rates $0 < epsilon << 1$ are introduced.
Since the adaptation of parenchymal cytokine communication is assumed to be slower than the immune counterpart @Sawicki2022DNM, it is chosen $epsilon^1 << epsilon^2 << 1$, which introduces dynamics on multiple timescales.

Lastly, the most influential parameter is $beta$ which controls the adaptivity of the cytokines.
Because $beta$ has such a big influence on the model dynamics it is called the _(biological) age parameter_ and summarizes multiple physiological concepts such as age, inflammatory baselines, adiposity, pre-existing illness, physical inactivity, nutritional influences and other common risk factors @Berner2022Critical.

All the systems variables and parameters are summarized in @tab:dnm together with their medical interpretation.
#figure(
  table(
    columns: (auto, auto, auto),
    // inset: 10pt,
    align: (center, left, left),
    table.header([*Symbol*], [*Name*], [*Physiological Meaning*]),

    table.cell(colspan: 3, align: left)[*Variables*],
    [$phi_i$], [Phase], [Group of cells],
    [$dot(phi)_i$], [Phase Velocity], [Metabolic activity],
    [$kappa_(i j)$], [Coupling Weight], [Cytokine activity],

    table.cell(colspan: 3, align: left)[*Parameters*],
    [$alpha$], [Phase lag], [Metabolic interaction delay],

    [$beta$],
    [Plasticity rule],
    [Combined of risk factors],
    // [Age, inflammation, pre-existing illness, other risk factor],

    [$omega$],
    [Natural frequency],
    [Natural cellular metabolism],

    [$epsilon$], [Time scale ratios], [Temporal scale of cytokine activity],
    // [$C$], [Initial network perturbation], [-],

    [$a_(i j)$],
    [Connectivity],
    [Fixed intra-organ cell-to-cell interaction],

    [$sigma$],
    [Interlayer coupling],
    [Interaction between parenchymal and \ immune cells through the basal lamina],

    table.cell(colspan: 3, align:left)[*Measures*],
    [$s$],
    [Standard deviation of frequency \ (see @eq:std)],
    [Pathogenicity (Parenchymal Layer)],
  ),
  caption: flex-caption(short: [#acl("DNM") Notation Table], long: [Summarization of notation used in the Dynamic Network Model. Superscripts indicating the layer are left out for readability.]),
) <tab:dnm>

=== Pathology in the DNM
A biological organism, such as the human body, can be regarded as a self-regulating system that, under healthy conditions, maintains a homeostatic state @Honan2021stroma.
Homeostasis refers to a dynamic but balanced equilibrium in which the physiological subsystems continuously interact to sustain stability despite external perturbations.
In the context of the #acr("DNM"), this healthy equilibrium state is represented by a synchronous regime of both layers in the duplex oscillator system.
In synchronous states, the organ layer and immune layer exhibit coordinated phase and frequency dynamics, reflecting balanced communication, collective frequency of cellular metabolism and stable systemic function.

Pathology, in contrast, is modeled by the breakdown of this synchronicity and the formation of frequency clusters in the parenchymal layer, i.e. loss of homeostatic balance.
In the #acr("DNM") at least one cluster will exhibit increased frequency and one with lower or unchanged frequency.
This aligns with medical observation, where unhealthy parenchymal cells change to a less efficient anaerobic glycosis-based metabolism, forcing them to increase their metabolic activity to keep up with the energy demand.
Remaining healthy cells are expected to stay frequency synchronized to a lower and "healthy" frequency.

There are two more cases, neither fully healthy nor fully pathologic, representing a vulnerable or resilient patient condition.
The healthy but vulnerable case corresponds to a splay state, where phases in the parenchymal layer are not synchronized, but the frequencies are, weakening the overall coherence @Berner2022Critical.
A resilient state corresponds to cases where both the phase and frequency of the parenchymal layer are synchronized, but the immune layer exhibits both frequency and phase clustering.

It is important to note, that the #acr("ODE") dynamics or system variable trajectories *do not* translate to the evolution of a patients pathological state.
Instead, the amount of desynchronization of the parenchymal layer when reaching a steady system state can be interpreted as the current state of a patients organ functionality.
The "result" or solution of the coupled oscillator system does not provide any temporal insights, but only describe the current condition.
Time-steps taken inside the model cannot be compared to any real-world time quantity. 

// #figure(tree_fig)

== Implementation <sec:dnmimp>
This subsection describes the implementation for the numerical integration of the #acr("DNM") defined in @eq:ode-sys, the choice of initial parameter values and how (de-)synchronicity/disease severity is quantified.
One goal of this implementation is to partly reproduce the numerical results presented in @Berner2022Critical, since they will be serving as a basis for following chapters.

=== Implementation Details
The backbone of the present numerical integration is JAX @jax2018, a Python package for high-performance array computation, similar to NumPy or MATLAB but designed for automatic differentiation, vectorization and #acr("JIT").
#acr("JIT")-compilation and vectorization allow high-level numerical code to be translated to highly optimized accelerator-specific machine code, for example #acr("GPU").
This way, performance benefits of massively parallel hardware can be utilized with minimal extra programming cost.
For the actual integration a differential equation solver from diffrax @kidger2022diffrax was used, which provides multiple solving schemes fully built on top of JAX.

While @Berner2022Critical uses a fourth-order Runge-Kutta method and a fixed step-size, this implementation#footnote[The code is available at https://github.com/unartig/sepsis_osc/blob/main/src/sepsis_osc/dnm/dynamic_network_model.py], in contrast, uses the Tsitouras 5/4 Runge-Kutta method @Tsitouras2011Runge with adaptive step-sizing controlled by a #acr("PID") controller, allowing for more efficient integration while keeping similar accuracy.
A relative tolerance of $10^(-3)$ and an absolute tolerance $10^(-6)$ were chosen.
All simulations were carried out in 64-bit floating point precision, necessary for accurate and stable system integration.

Because of the element-wise differences used in the coupling terms $phi^ot_i-phi^ot_j in RR^(N times N)$ the computational cost scales quadratically with the number of oscillators $N$.
These differences are then transformed by the computationally expensive trigonometric $sin$ routine.
To accelerate integration, these trigonometric evaluations were optimized following @KuramotoComp.
Terms of the form $sin(theta_l-theta_m)$ were expanded as:
$
  sin(theta_l-theta_m)=sin(theta_l)cos(theta_m) - cos(theta_l)sin(theta_m) "    " forall l,m in {1,...,N}
$
By caching the terms $sin(theta_l)$, $sin(theta_m)$, $cos(theta_l)$, $cos(theta_m)$ once per iteration, the number of trigonometric evaluations per iteration is reduced from $2dot[N (N-1)]$ to $2dot[4N]$, significantly improving performance for mid to large oscillator populations.

Additionally, an alternative implementation based on Lie-algebra formulations was also explored, utilizing their natural representation for rotations in N-D-space.
Although theoretically promising in terms of numerical accuracy and integration stability, this approach did not yield practical advantages in performance.
Further details on this reformulation are provided in @a:lie.

=== Parameterization and Initialization <sec:init>
The #acr("DNM") is dimensionless and not bound to any physical scale, that means there is no medical ground truth of parameter values and their choice is somewhat arbitrary.
For the present implementation the parameterization is adopted from the original works @Sawicki2022DNM and @Berner2022Critical since they have already shown desired properties of (de-)synchronization and valid medical interpretations for these parameter choices.

The majority of their parameter choices heavily simplify the model.
First of all, the different natural frequencies are treated as equal and are set to $0$, giving $omega^1 = omega^2 = omega = 0$, any other choice of $omega$ just changes the frame of reference (co-rotating frame), the dynamics stay unchanged @Berner2022Critical.
The phase lag parameters for the inter layer coupling are both set to $alpha^(1 2) = alpha^(2 1) = 0$, yielding instantaneous interactions, the intralayer phase lags are set to $alpha^11 = alpha^22 = -0.28pi$, which was the prominently used configuration in @Berner2022Critical yielding the desired dynamical properties.
The constant intralayer coupling in the parenchymal layer is chosen as global coupling $a_(i j) = 1 " if " i!=j " else " 0$.

The adaptation rates are chosen as $epsilon^1=0.03$ and $epsilon^2=0.3$, creating the two dynamical timescales for slow parenchymal and faster immune cells.
The number of oscillators per layer is chosen as $N=200$ throughout all simulations.
To account for the randomly initialized variables, each parameter configuration is integrated for an ensemble of $M=50$ initializations.
This randomization of initial values is used to account for epistemic uncertainties, i.e. systemic errors introduced by modeling simplifications.

In @Berner2022Critical the influence of parameter values for $beta$ and $sigma$ was investigated and not constant throughout different simulations, with $beta in [0.4pi, 0.7pi]$ and $sigma in [0, 1.5]$, in this work the interval for $beta$ was increased to $[0.0, 1.0pi]$.
An exhaustive summary of all variable initializations and parameter choices can be found in @tab:init.

#figure(
  table(
    columns: (auto, 13em, auto, 13em),
    align: center,
    table.header([*Symbol*], [*Value*], [*Symbol*], [*Value*]),
    table.cell(colspan: 4, align: left)[*Variables*],
    [$phi^1_i$], [$~cal(U)[0, 2pi)$],
    [$kappa^1_(i != j)$],
    [$~cal(U)(-1, 1)$],
    [$phi^2_i$], [$~cal(U)[0, 2pi)$],
    [$kappa^2_(i != j)$],
    [clusters of size $C$ and $1-C$],

    table.cell(colspan: 4, align: left)[*Parameters*],
    [$M$], [50], [$N$], [200],
    [$C$], [$20%$], [], [],
    [$beta$], [$[0.0, 1.0]pi$], [$sigma$], [$[0.0, 1.5]$],
    [$alpha^11, alpha^22$], [$-0.28pi$], [$alpha^12, alpha^21$], [0.0],
    [$omega_1, omega_2$], [0.0], [$A^1$], [$bb(1) - I$],
    [$epsilon^1$], [0.03], [$epsilon^2$], [0.3],
  ),
  caption: flex-caption(
  short: [Simulation Parameterization],
  long: [Parameterization and initialization of the #acr("DNM") used for the numerical integration.]),
)<tab:init>

Initial values for the system variables, i.e. the phases and coupling strengths, were not parametrized explicitly, rather sampled from probability distributions.
The initial phases $phi(0)^ot_i$ are randomly and uniformly distributed around the unit circle for both layers, i.e. $phi(0)^ot_i ~ cal(U)[0, 2pi)$.
The intralayer coupling of the parenchymal layer coupling is also chosen randomly and uniformly distributed in the interval $[-1.0, 1.0]$.
Since there is no self-coupling, the diagonal of the intralayer coupling matrix is set to 0 $kappa^ot_(i j) = 0$.

For the immune layer an initial cytokine activation is modeled by clustering the initial intralayer coupling matrix.
A smaller cluster of $C dot N$ oscillators and a bigger cluster of $(1-C) dot N$ cells.
Within the clusters oscillators are connected but not between the clusters.
Following @Berner2022Critical, the cluster size $C in [0, 0.5]$ was chosen as 0.2, but as their findings suggest the size of the initial clusters does not have impact on the systems dynamics.
// Simulations have shown that even without any clustering, meaning $bold(Kappa)^2=bb(0)$ or $bold(Kappa)^2=bb(1)$, the dynamics stay unchanged, making this initialization choice meaning-free, it is stated here just for completeness.
An example for initial variable values of a system with $N=200$ and $C=0.2$ is shown in @fig:init.

#figure(
  image("../images/init.svg", width: 100%),
  caption:
  flex-caption(
  short: [#acl("DNM") Initialization],
  long: [
    Initializations for the variable values of a #acr("DNM") with $N=200$ oscillators per layer.
    The middle two plots show the phases of the oscillators, with $phi^1_i$ for parenchymal and $phi^2_i$ for the immune layer, sampled from a uniform random distribution from 0 to $2pi$.
    On the left-hand side is the initialization of the parenchymal intralayer coupling matrix $bold(Kappa)^1$ from a uniform distribution in the interval from -1 to 1.
    On the right-hand side is the two cluster initialization for the coupling matrix $bold(Kappa)^2$ of the immune layer, with a cluster size of $C=0.2$, where each cluster is intra-connected, but without connections between the clusters.
  ]),
) <fig:init>

=== Synchronicity Metrics
As introduced in @sec:kuramoto, for the Kuramoto networks the synchronization behavior is usually the point of interest, in the following two metrics are introduced, relevant to connect the #acr("DNM")-dynamics to sepsis.
There are two relevant states or system configurations that should be identifiable and quantifiable to allow qualified state analyzes: phase and frequency synchronization, for each a distinct measure is required.

*Phase synchronization* of a layer is commonly measured by the _Kuramoto Order Parameter_ @acebron2005kuramoto:

$
  R^ot_2 (t) = 1/N abs(sum^N_j e^(i dot phi^ot_j (t))) "   with " 0<=R^ot_2<=1
$
where $R^ot_2=0$ corresponds to total desynchronization, the splay-state and $R^ot_2=1$ corresponds to fully synchronized state, for convenience from now on the subscript $dot_2$ is omitted, denoting the Kuramoto Order Parameter simply as $R^ot$.

*Frequency synchronization* measurements are more involved, as a starting point first the notion of a layers _mean phase velocity_ has to be introduced, which can be calculated as follows:

$
  overline(omega)^ot (t)= 1/N sum^N_j dot(phi)^ot_j (t)
$ <eq:mean>
The original definition uses an approximated version using the oscillators mean velocity:
$
  quad "  " tilde(mean(dot(phi)^ot_j (t))) & = (phi^ot_j (t + T) - phi^ot_j (t))/T
$$
   tilde(overline(omega)^ot (t)) & = 1/N sum^N_j tilde(mean(dot(phi)^ot_j (t)))
$
for some averaging time window $T$.
But since their choice of $T$ is not documented while having substantial influence on the calculation the instantaneous angle velocity from @eq:mean was preferred.
While the overall systematics stay the same, deviations from their original results are expected, due to this change in definition.

One can now calculate the _standard deviation of the mean phase velocities_:
$
  sigma_chi (overline(omega)^ot, t) = sqrt(1/N sum^N_j (dot(phi)^ot_j (t) - overline(omega)^ot (t))^2)
$ <eq:stdsingle>
Where $sigma_chi = 0$ indicates full frequency synchronization and growing values indicate desynchronization and/or clustering.
Nonzero values only reveal that there is some desynchronization of the frequencies, but it remains unknown if it is clustered, multi-clustered or fully desynchronized.

Having multiple ensemble members $m=1,...,M$ with the same parameterization, it is expected that different initializations, meaning initial-values drawn from the parameterized distributions, exhibit dissimilar behaviors, one can also calculate the _ensemble averaged standard deviation of the mean phase velocity_:

$
  s^ot (t) = 1/M sum^M_m sigma_chi (overline(omega)_m^ot, t)
$ <eq:std>
In @Berner2022Critical it was shown numerically that the quantity $s^ot$ is proportional to the fraction of ensemble members that exhibit frequency clusters containing at least one oscillator.
This makes $s^1$ a viable measure for pathology, as increasing values of $s^1$ or increasing system incoherence then indicate more dysregulated host responses and consequently higher risks of multiple organ failure.

=== Simulation Results <sec:dnmres>
The original findings of @Berner2022Critical identify $beta$, the combined age parameter, and $sigma$, the interlayer coupling strength which models the communication between organ and immune cells, as naturally important parameters in order to understand underlying mechanisms of sepsis progression.
In the following subsection multiple simulation results are presented, starting with time-snapshots for different parameterizations and initializations.
Afterwards, the transient and temporal behavior of the metrics $s^ot$ and $R^ot$ is presented for the same parameterization, as well as the introduction of the $beta, sigma$ phase space of these metrics.

In @fig:snap, snapshots of the system variables are shown for different parameterization, differing only in the choice $beta$ and $sigma$, configurations A, B, C and D are listed in @tab:siminit, other parameters are shared between the configurations and are stated in @tab:init.
Each configuration is expected to represent the current physiological state of an individual patient.

All the following results are for a system with $N=200$ oscillators, and snapshots taken at time $T_"sim"=2000$, the end of the integration time, and show the stationary values at that time point.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    table.header([], [*A*], [*B*], [*C*], [*D*]),
    [$beta$], [$0.5 pi$], [$0.58 pi$], [$0.7 pi$], [$0.5 pi$],
    [$sigma$], [$1.0$], [$1.0$], [$1.0$], [$0.2$],
  ),
  caption: flex-caption(short: [Specific $beta$-$sigma$ combinations to illustrate simulation results], long:[Specific $beta$-$sigma$ combinations to illustrate simulation results.]),
)<tab:siminit>

In @fig:snap, the left-most columns depicts the coupling matrices for the organ layer $bold(Kappa)^1$ followed by two columns showing the phase velocities for each oscillator $dot(phi)_i^ot$ and two columns showing the oscillator phases each layer $phi_i^ot (T_"sim")$.
The right-most column shows the coupling matrix for the immune layer $bold(Kappa)^2$.
Individual oscillators of each layer are sorted first from lowest to highest frequency and secondary by lowest to highest phase for better clarity.
@fig:snap/C and @fig:snap/C' share the same parameterization but are different samples from the same initialization distributions.

#figure(
  image("../images/snapshots.svg", width: 100%),
  caption: flex-caption(
  short: [Snapshots of simulated system states],
  long: [
    Snapshots of different #acr("DNM") parametrization and initialization. Configuration *A* can be regarded as healthy, with phases and frequencies being fully synchronized.
    In contrast, *B* and *C* are pathologic, due to their clustering in $dot(phi)^1$. Configuration *C'* corresponds to a vulnerable state, because of uniformly distributed phases (splay state).
    Lastly, *D* is regarded as resilient, since the phases exhibit clustering, but the frequencies $dot(phi)^1$ are synchronized.
  ]),
) <fig:snap>
@fig:snap/A is fully synchronized/coherent since it not only has the frequencies synchronized but also the phases and can therefore interpreted as healthy.
The coherence can also be seen in the fully homogeneous coupling matrices where both $bold(Kappa)^ot$ show the same coupling strength for every oscillator pair.
@fig:snap/B and @fig:snap/C in contrast show signs of a pathological state, here both the frequencies and phases have distinct clusters.
The clusters are also slightly visible in the coupling matrices, where the coupling strength differs based on the cluster, with the smaller clustering having exhibiting a weaker coupling.
@fig:snap/C', even though having the same parameterization as @fig:snap/C, can be regarded vulnerable, since the phases uniformly distribute in the $[0, 2pi)$ interval ($R^ot = 0$), while the frequencies are synchronized.
Coupling matrices for @fig:snap/C' show traveling waves, which are characteristic for splay states.
Observing different results for different initializations justifies the introduction of ensembles.
Lastly row @fig:snap/D shows a resilient state, where the phases are clustered while the frequencies are synchronized.

In @fig:ensemble, where the ensembles were introduced, every configuration of A, B, C, and D was simulated for $M=50$ different initializations over an interval of $T_"sim"=2000$.
The two left columns show the standard deviation of the mean phase velocities $sigma_chi$ for each ensemble member $m in 1,2...M$ (blue) and the aggregated ensemble mean $s^ot$ (orange).
The plots show the temporal evolution of metrics for quantifying phase and frequency coherence, with the two right-most columns of @fig:ensemble show the temporal behavior of the Kuramoto Order Parameter for each individual ensemble member $m$ (blue) and the aggregated ensemble mean (orange).
Where lower values for $R^ot$ indicate decoherence in phases, with its minimum $R^ot = 0$ coincides with a splay state, and for $s^ot$ higher values indicate a larger amount of frequency decoherence and clustering.
#figure(
  image("../images/ensembles.svg", width: 100%),
  caption:
  flex-caption(
  short: [Temporal evolution of the phase- and frequency-synchronization metrics],
  long: [Transient and temporal evolution of the phase- and frequency-synchronization metrics $R^ot$ and $s^ot$, for ensembles of the #acr("DNM") for the configurations listed in @tab:siminit.
    Emphasizing the influence of $beta$ and $sigma$ on the systems synchronization behavior, and presenting different stable emergent system states.
    Single ensemble members are shown in blue and the ensemble mean in orange.
  ]),
) <fig:ensemble>

Every ensemble in @fig:ensemble shows decoherence for early time-points, which is expected for randomly initialized variables, but changes quickly through a transient phase $t in [0.0, 200]$ into systematic stable behavior.
In @fig:ensemble/A, the stable states align with the observations made for @fig:snap/A, besides small jitter the majority of ensemble members exhibits synchronized frequencies $s^ot approx 0$.
Also the phases of configuration @fig:ensemble/A are mostly synchronized with the ensemble mean $overline(R)^ot approx 1$, i.e. $overline(R)^ot = 1/M sum^M_m R^ot_m$ with $M=50$, only two initializations (ensemble members) show decoherence and are oscillating between weak clustering and almost full incoherence.
Medically this can be interpreted as a low risk of a dysregulated host response for an otherwise healthy response to the initial cytokine activation.
For configuration @fig:ensemble/B, the amount of incoherence inside the ensemble is clearly visible, with mean $s^ot$ being positive and some more ensemble members exhibiting clustering, indicated by a Kuramoto Order Parameter slightly less than $1$.
In configuration @fig:ensemble/C the incoherence is even more prominent, larger $s^ot$ and some ensemble members evolving into a splay state, i.e. $R^ot=0$.
For configuration @fig:ensemble/D the overall phase incoherence is again a bit less compared to @fig:ensemble/C, and lower for the organ compared to the immune layer.
The phases are coherent for the organ layer but seem almost chaotic for the immune layer, some are synchronized, while others are clustered, in a chimera or almost splay-like state.
Over the whole simulation period, the coherency in the immune layer seems not to fully stabilize, rather oscillate around an attractor.

Each of the configurations only differs in the parameter choices for $beta$ and $sigma$, yet they evolve into unique and distinct system states.
To put these four specific configurations into broader perspective, a grid of $beta$ and $sigma$ was simulated, in the interval $beta in [0, 1]$ with a grid resolution of $0.01$ and $sigma in [0, 1.5]$ with a resolution of $0.015$, creating a grid of $10,000$ points.
In @fig:phase the metric $s^ot$ is shown in the $beta-sigma$ phase space for both layers in the first row, where brighter colors indicate a larger risk of frequency desynchronization, or risk of dysregulated immune response.
The second row shows the ensemble mean over $overline(R)^ot$ where darker colors indicate larger phase desynchronization.
The white rectangle indicates the simulated region in @Berner2022Critical, $beta in [0.4, 0.7]$ and $sigma in [0, 1.5]$ for reference.
Coordinates of configurations A, B, C, and D are also labeled.
#figure(
  image("../images/phase.svg", width: 100%),
  caption: flex-caption(
  short: [Phase Space of $sigma$ and $beta$],
  long: [
    The plots show the phase space of the parameters $beta$ and $sigma$ and illustrate the broader picture their influence on the frequency and phase synchronization of the #acr("DNM").
    White rectangle indicates the grid-limits of the original publication @Berner2022Critical
  .
  ]),
) <fig:phase>
Generally, there is a similarity between phase and frequency desynchronization but no full equality, meaning there are parameter regions where the phase is synchronized and frequency desynchronized and vice versa.
Another observation, that smaller values of $beta < 0.55$ correspond to less desynchronization and stronger coherence, which is in line with the medical interpretation of $beta$ where smaller values indicate a younger and more healthy biological age.
When crossing a critical value of $beta_c approx 0.55$ for the frequency and $beta_c approx 0.6$ for the phases, the synchronization behavior suddenly changes and tends towards incoherence, clustering and pathological interpretations.

For small values of $sigma < 0.5$ the frequency synchronization and $sigma < 0.25$ for the phase synchronization, the behavior significantly differs between immune and organ layer.
The immune layer tends to fully desynchronize, instead the organ layer only the frequency desynchronizes for larger $beta > 0.7$.
With larger values of $sigma > 0.5$ the dynamics more or less harmonize between layers and metrics and are mostly depend on $beta$.

== Limitations and Research Direction <sec:problemdef>
While the #acr("DNM") provides a structured, functional and biologically inspired model sepsis for dynamics, several limitations must be acknowledged when interpreting its physiological relevance.

The fully connected intralayer topology (all oscillators inside a layer interact with each other), may not reflect actual cytokine networks behavior, where communication is spatially localized.
Similarly the interlayer is modeled via uniform connections which treats infected areas not different than healthy areas, even though the communication patterns are most likely not the same for both cases.

In the original work, and adapted for this thesis, only a very small subspace has been simulated and interpreted @Berner2022Critical.
For this selected subspace one can find regimes of synchronization, phase clustering and frequency clustering, which allow physiological interpretation.
But what about the unexplored space, does it not align with existing medical knowledge?

Central parameters like $beta$ the medical age, $sigma$ the interlayer coupling, the phase lags $alpha$, and the timescale ratios $epsilon$ do not correlate to any real observable quantity.
This dimensionless formulation makes the model scale-free, which strengthens theoretical analysis but possibly weakens clinical translation.

The #acr("DNM") is explicitly functional, where individual oscillators do not correspond to specific cells, instead the model captures aggregated behavior.
This abstraction brings the benefit of computational tractability, compared to mechanistic cell simulation, and simplified system state interpretation, but it also brings drawbacks of limited ability to identify which biological processes drive the observed dynamics.
It boils down to the question: Does the functional description capture the essential dynamics of sepsis, or does it produce patterns that resemble physiological behavior without true correspondence?

These limitations reveal a gap: while the #acr("DNM") demonstrates rich theoretical behavior that can be interpreted through a physiological lens, the model has not yet been validated against real patient data.
To bridge this gap between theoretical considerations and clinical utility, this work investigates whether the #acr("DNM") can be grounded in observable patient trajectories.
Specifically, can the abstract parameters ($beta$, $sigma$) that govern the model behavior be inferred from real-world time-series data?
And if so, does incorporating this physics-inspired structure provide advantages over purely data-driven approaches?

To summarize, the specific research questions include:
#quote(block: true)[
  _*1) Usability of the #acr("DNM")*: How and to what extent can #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states?_

  _*2) Comparison with data-based approaches*: How can the model-based predictions be compared with those of purely data-based approaches in terms of predictive power?_
]

The first question directly addresses the parameter interpretability problem.
If $beta$ and $sigma$ cannot be reliably inferred from clinical observations, or if inferred values lack predictive power for outcomes, the models clinical relevance remains speculative.
Conversely, if #acr("DNM") parameters learned from data correlate with disease severity and trajectory, this would suggest physiological correspondence to the model parameters.

The second question tackles the functional vs. mechanistic limitation.
Does embedding the #acr("DNM") structure as an inductive bias improve predictions compared to black-box models that ignore physiological principles? Or does the model's abstraction level sacrifice too much biological detail to be useful? This comparison will reveal whether the DNM's theoretical framework translates to practical advantages in clinical prediction tasks.

== Summary of the Dynamic Network Model
This chapter introduced the #acl("DNM") as a functional, mesoscopic description of coordinated physiological activity during sepsis, modeling cellular cytokine-based communication.
Based on adaptive Kuramoto-type oscillators arranged in a two-layer parenchymal and immune architecture, the model exhibits a range of emergent regimes like synchronization, clustering, chimera-like patterns, that may be interpreted physiological states.
Key parameters such as the biological age $beta$ and interlayer coupling $sigma$ were shown to modulate these regimes.
 However, the gap between theoretical considerations and clinical validation remains unaddressed.
 The following chapters address this gap directly by developing methods to infer #acr("DNM") parameters from real patient data and evaluating whether this physics-based structure provides predictive advantages over purely data-driven and model free approaches.
