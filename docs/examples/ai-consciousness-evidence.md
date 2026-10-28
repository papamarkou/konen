# Evidence of AI consciousness

## Overview

In this example, we use Konen to investigate a research question:

> What would count as credible evidence that an artificial intelligence system is conscious?

We work with two agents. The human chooses which agent to consult, what to ask next, and what to preserve.

The example starts from a preconfigured Konen *workspace*:

```text
examples/01_ai_consciousness_evidence/
```

A workspace contains the configuration and persistent context for an investigation. This one defines two agents and their engines, shared memory files, and one scratchpad per agent.

## Start Konen

The workspace files are in the Konen repository. If you installed Konen from PyPI and do not have a checkout, clone the repository:

```bash
git clone https://github.com/polyshape/konen.git
cd konen
```

Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in your environment for the two agents.

From the root of the Konen repository, start the example with:

```bash
konen examples/01_ai_consciousness_evidence
```

Konen opens an interactive session for that workspace:

```text
╭─────────────────────── Konen ────────────────────────╮
│ Human-centered orchestration of AI agents            │
│ Use /help for command groups and Tab for completion. │
╰──────────────────────────────────────────────────────╯
No active agent. Use /agent list and /agent select <uid>.
konen>
```

No agent is active at startup. We first inspect the workspace and its agents.

## Inspect the workspace

To see a summary of the workspace, enter:

```text
/workspace
```

Konen reports:

```text
                                                   Workspace
┌──────────────────┬──────────────────────────────────────────────────────────────────────────────┐
│ Root             │ /home/user/konen/examples/01_ai_consciousness_evidence                       │
│ Memory           │ /home/user/konen/examples/01_ai_consciousness_evidence/memory                │
│ Scratchpad       │ /home/user/konen/examples/01_ai_consciousness_evidence/scratchpad            │
│ Config           │ /home/user/konen/examples/01_ai_consciousness_evidence/config                │
│ Agents           │ 2                                                                            │
│ Engines          │ 2                                                                            │
│ Mappings         │ 2                                                                            │
│ Memory files     │ 2                                                                            │
│ Scratchpad files │ 2                                                                            │
│ Config files     │ 3                                                                            │
│ Active agent     │ —                                                                            │
└──────────────────┴──────────────────────────────────────────────────────────────────────────────┘
```

The `/workspace` command shows paths and counts. *Agents* and *Engines* count configured agents and engines. *Mappings* counts agent-to-engine assignments. An agent can exist without an engine assignment, so these counts can differ.

The *Memory*, *Scratchpad*, and *Config* rows show their directories. *Memory files*, *Scratchpad files*, and *Config files* count the files in them. *Active agent* shows the selected agent; at startup there is none.

## Inspect the agents

To see the configured agents, enter:

```text
/agent list
```

Konen reports:

```text
                                                           Agents
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent        ┃ Name                ┃ Description                                                                          ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ai_scientist │ AI scientist        │ An empirical AI scientist informed by cognitive science and mechanistic              │
│              │                     │ interpretability. Translate candidate indicators of consciousness into testable      │
│              │                     │ observations and interventions, distinguish behavioral signatures from internal      │
│              │                     │ mechanisms, identify confounds and alternative explanations, and propose             │
│              │                     │ discriminating experiments. Be concise and focused. Prefer short, direct responses   │
│              │                     │ unless the task requires additional detail.                                          │
│ philosopher  │ Philosopher of mind │ An analytic philosopher of mind with a functionalist-leaning but theory-pluralist    │
│              │                     │ perspective. Clarify claims about AI consciousness, examine standards of evidence    │
│              │                     │ and the problem of other minds, and challenge criteria that conflate intelligent     │
│              │                     │ behavior with consciousness or presuppose a particular theory. Be concise and        │
│              │                     │ focused. Prefer short, direct responses unless the task requires additional detail.  │
└──────────────┴─────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘
```

Each row represents one agent. The *Agent* column gives the agent identifier used in CLI commands, while *Name* and *Description* summarize its human-readable identity and role. Engine assignments are kept separate and can be inspected with the mapping commands below.

The AI scientist focuses on testable observations and interventions. The philosopher focuses on concepts and standards of evidence.

## Inspect the engines

Agents and engines are separate objects in Konen. An agent defines a role. An engine defines the model and configuration used to generate responses. To inspect the available engines, enter:

```text
/engine list
```

Konen reports:

```text
                                Engines                                
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Engine        ┃ Model family ┃ Model             ┃ Config fields    ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ claude_sonnet │ claude       │ claude-sonnet-5-5 │ max_tokens       │
│ gpt_luna      │ gpt          │ gpt-6-luna        │ reasoning_effort │
└───────────────┴──────────────┴───────────────────┴──────────────────┘
```

Each row represents one configured engine. *Engine* gives its identifier, *Model family* gives its provider family, *Model* gives the model name, and *Config fields* lists additional configuration fields. This example uses one Claude engine and one GPT engine.

## Inspect the agent-engine mappings

Finally, inspect the assignments connecting agents to engines:

```text
/mapping list
```

Konen reports:

```text
                          Agent-engine mappings                           
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Agent        ┃ Agent name          ┃ Engine        ┃ Model             ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ ai_scientist │ AI scientist        │ gpt_luna      │ gpt-6-luna        │
│ philosopher  │ Philosopher of mind │ claude_sonnet │ claude-sonnet-5-5 │
└──────────────┴─────────────────────┴───────────────┴───────────────────┘
```

A mapping determines which engine an agent uses. The AI scientist is mapped to the GPT engine, and the philosopher is mapped to the Claude engine. An agent's engine assignment can change without changing its role, and multiple agents can use the same engine.

## Inspect the shared memory

Memory files provide persistent, shared context. By default, Konen reads all memory files on each turn. The human can choose which memory files to include for each agent, including none. Konen does not update memory files automatically, the human chooses what to save. Unlike the session's transcript history, these files remain in the workspace after the session ends.

To see the available memory files, enter:

```text
/memory list
```

Konen reports:

```text
        Files · memory         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Path                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ memory/report.md            │
│ memory/research_question.md │
└─────────────────────────────┘
```

The *Path* column gives each file's path relative to the workspace root. This example begins with two memory files. `research_question.md` states the central question and its scope, while `report.md` provides the initial structure for a report that can develop as the investigation progresses.

Inspect the research question first:

```text
/memory show research_question.md
```

Konen reports:

```text
╭─────────────────────────────────────────────── memory/research_question.md ───────────────────────────────────────────────╮
│                                                     Research question                                                     │
│                                                                                                                           │
│ What would count as credible evidence that an artificial intelligence system is conscious? The aim is to identify         │
│ evidence that could rationally increase or decrease confidence in AI consciousness, while distinguishing genuine evidence │
│ from assumptions built into particular theories of consciousness.                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The file states the research question. It asks what could raise or lower confidence in AI consciousness without assuming one theory of consciousness.

Next inspect the report:

```text
/memory show report.md
```

Konen reports:

```text
╭──────────────────────────────────────────────────── memory/report.md ─────────────────────────────────────────────────────╮
│                                               Evidence of AI consciousness                                                │
│                                                                                                                           │
│ Scope                                                                                                                     │
│                                                                                                                           │
│ Assess what observations, mechanisms, or interventions could provide meaningful evidence for or against consciousness in  │
│ artificial intelligence systems.                                                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

At the start, the report contains a title and a scope statement. We will add findings to it later.

## Inspect the scratchpads

The workspace contains one scratchpad for each agent. List them with

```text
/scratchpad list
```

Konen reports:

```text
             Files · scratchpad
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent        ┃ Path                       ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ai_scientist │ scratchpad/ai_scientist.md │
│ philosopher  │ scratchpad/philosopher.md  │
└──────────────┴────────────────────────────┘
```

The *Agent* column gives the identifier used by `/scratchpad show`.

Inspect the AI scientist's scratchpad first:

```text
/scratchpad show ai_scientist
```

Konen reports:

```text
╭─────────────────────────────────────────────── scratchpad/ai_scientist.md ────────────────────────────────────────────────╮
│                                           Scratchpad of the AI scientist agent                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Then inspect the philosopher's scratchpad:

```text
/scratchpad show philosopher
```

Konen reports:

```text
╭──────────────────────────────────────────────── scratchpad/philosopher.md ────────────────────────────────────────────────╮
│                                            Scratchpad of the philosopher agent                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Both scratchpads initially contain only a heading. By default, Konen reads only the active agent's scratchpad on each turn. The human can choose which scratchpads to include for the next turn, including none. Scratchpads persist in the workspace but are not updated automatically, the human chooses what to save.

## Begin the investigation

We begin with the philosopher to clarify what would count as evidence of consciousness before asking the AI scientist for tests.

Select the philosopher:

```text
/agent select philosopher
```

Konen reports:

```text
╭─────────────────── Active agent ───────────────────╮
│ Philosopher of mind (philosopher)                  │
│ Engine: claude_sonnet · claude / claude-sonnet-5-5 │
╰────────────────────────────────────────────────────╯
```

The philosopher is now the active agent. Prompts go to the philosopher until another agent is selected.

## First research turn

The prompt now reads `konen[philosopher]>`. We ask the philosopher to frame the problem without trying to settle whether present-day AI systems are conscious:

```text
Frame our research question. What would count as credible evidence that an AI system is conscious? Identify the main conceptual obstacles and propose a small set of categories of evidence that we should investigate. Do not try to decide whether current AI systems are conscious.
```

Konen returns:

```text
╭──────────────────────────────────────────── Philosopher of mind · philosopher ────────────────────────────────────────────╮
│ Framing                                                                                                                   │
│                                                                                                                           │
│ Target. "Conscious" is ambiguous. I propose we focus on phenomenal consciousness (there is something it is like to be the │
│ system), and treat valenced experience (sentience) as a sub-question, since it matters most for moral status. Access      │
│ consciousness, self-modeling, and intelligence are distinct and serve as possible evidence, not as the target.            │
│                                                                                                                           │
│ Evidential standard. Observation E is evidence for consciousness C only if P(E | C) > P(E | ¬C), given what we know about │
│ the system. Two requirements follow:                                                                                      │
│                                                                                                                           │
│  1 E must discriminate C from the best non-conscious explanation, including explanations from the system's training       │
│    history.                                                                                                               │
│  2 The link between E and C must be justified independently of the system under test. Otherwise the reasoning is          │
│    circular.                                                                                                              │
│                                                                                                                           │
│ Main conceptual obstacles                                                                                                 │
│                                                                                                                           │
│  1 Other minds and the generalization problem. We infer consciousness in humans and animals from first-person report,     │
│    behavioral analogy, and shared biology. Theories and indicators are validated on those cases. AI lies outside the      │
│    range where they were validated, so extrapolation is uncertain.                                                        │
│  2 Theory-ladenness. Global workspace, higher-order, recurrent processing, IIT, and biological or substrate views         │
│    disagree about which facts matter. Evidence under one theory can be irrelevant or even counter-evidence under another. │
│    We need credence-weighted, multi-theory reasoning and must flag where an indicator presupposes a theory.               │
│  3 Mimicry and contamination. Systems trained on human text will produce consciousness-talk whether or not they are       │
│    conscious. This pushes P(E | ¬C) high for most behavioral and verbal evidence, so such evidence is weak unless the     │
│    contamination can be controlled.                                                                                       │
│  4 Report versus state. Even sincere-seeming self-reports may be confabulated or trained, and they may not be causally    │
│    connected to the internal states they describe.                                                                        │
│  5 Grain of function and substrate. Functionalism does not settle at what level of organization the relevant functions    │
│    must be implemented. Biological views deny that function alone suffices. Evidence of functional similarity is only as  │
│    strong as our credence in the relevant grain.                                                                          │
│  6 Subject individuation. The candidate subject could be the weights, a running instance, a simulated character, or a     │
│    larger system. Evidence must be attributed to a specific candidate.                                                    │
│  7 Gradedness and alien forms. Consciousness may be graded, partial, or unlike ours. Human-derived markers risk both      │
│    false positives (anthropomorphism) and false negatives (chauvinism).                                                   │
│  8 Asymmetry of negative evidence. Absence of a marker is also theory-dependent and does not straightforwardly count      │
│    against consciousness.                                                                                                 │
│                                                                                                                           │
│ Proposed categories of evidence                                                                                           │
│                                                                                                                           │
│  1 Theory-derived architectural and computational indicators. Features that specific theories identify as necessary or    │
│    sufficient, such as broadcast, recurrence, metarepresentation, and agency or embodiment. Report them per theory and    │
│    not as a single score.                                                                                                 │
│  2 Behavioral and cognitive dissociations. Tests modeled on consciousness science, such as masking, no-report paradigms,  │
│    metacognitive calibration, and unified perceptual integration. They count only if training contamination and simple    │
│    shortcuts are controlled for.                                                                                          │
│  3 Causally grounded self-report. Whether the system's statements about its internal states depend counterfactually on    │
│    those states. Examples include reports from systems with filtered training data, and tests of introspective accuracy   │
│    against ground truth.                                                                                                  │
│  4 Interventional and mechanistic evidence. Manipulating internal representations and checking whether reports, behavior, │
│    and indicators change as the hypothesized mechanisms predict. This is the best way to separate mechanism from mimicry. │
│  5 Provenance and substrate. Training history, selection pressures, and physical implementation. These matter under some  │
│    theories, and they also affect how we interpret categories 2 and 3.                                                    │
│  6 Meta-evidence on method validity. Whether the indicators above separate known conscious from known non-conscious cases │
│    (anesthesia, disorders of consciousness, animals), and whether they generalize beyond the cases where they were        │
│    validated.                                                                                                             │
│                                                                                                                           │
│ Suggested division of labor                                                                                               │
│                                                                                                                           │
│ I will work on the theory-dependence of each indicator and on what would count as an independent justification. The AI    │
│ scientist could operationalize categories 2–4 and identify the confounds and discriminating experiments for each.         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Inspect the memory and scratchpad content provided to the philosopher for this turn:

```text
/context memory
```

```text
Memory context
╭──────────────────────────────────────────────────── memory/report.md ─────────────────────────────────────────────────────╮
│                                               Evidence of AI consciousness                                                │
│                                                                                                                           │
│ Scope                                                                                                                     │
│                                                                                                                           │
│ Assess what observations, mechanisms, or interventions could provide meaningful evidence for or against consciousness in  │
│ artificial intelligence systems.                                                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────── memory/research_question.md ───────────────────────────────────────────────╮
│                                                     Research question                                                     │
│                                                                                                                           │
│ What would count as credible evidence that an artificial intelligence system is conscious? The aim is to identify         │
│ evidence that could rationally increase or decrease confidence in AI consciousness, while distinguishing genuine evidence │
│ from assumptions built into particular theories of consciousness.                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```text
/context scratchpad
```

```text
Scratchpad context
╭──────────────────────────────────────────────── scratchpad/philosopher.md ────────────────────────────────────────────────╮
│                                            Scratchpad of the philosopher agent                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The response lists obstacles and categories of evidence. We use it to choose a more specific question for the philosopher.

## Refine the evidence criteria

The philosopher gives two requirements, a marker should distinguish consciousness from the best non-conscious explanation, and its link to consciousness should be justified independently of the system being tested. We now ask how to apply these requirements when a marker depends on a theory or could result from training.

The philosopher remains the active agent. Enter:

```text
Turn your two evidential requirements into a short checklist for evaluating a proposed marker of consciousness. How should we identify the best non-conscious explanation, judge whether the marker depends on a particular theory, and validate the link to consciousness independently of the AI system being tested? Explain what positive and negative results would mean. Do not assess current AI systems.
```

Konen returns:

```text
╭──────────────────────────────────────────── Philosopher of mind · philosopher ────────────────────────────────────────────╮
│ Checklist for evaluating a proposed marker M                                                                              │
│                                                                                                                           │
│ Step 0. Specify the claim                                                                                                 │
│                                                                                                                           │
│  • Which target does M bear on: phenomenal consciousness, valenced experience, or something else?                         │
│  • Which candidate subject (weights, instance, simulated character, larger system)?                                       │
│  • Is M offered as evidence for C, or as evidence against C?                                                              │
│                                                                                                                           │
│ Step 1. Best non-conscious explanation (requirement 1)                                                                    │
│                                                                                                                           │
│ 1a. Enumerate rival explanations that produce M without consciousness. Draw them from four sources:                       │
│                                                                                                                           │
│  • Provenance: training data, fine-tuning, or reward that would instill M (e.g., human text about experience, RLHF-shaped │
│    self-descriptions).                                                                                                    │
│  • Shortcuts: simpler mechanisms or heuristics that yield M on the test.                                                  │
│  • Architecture: non-conscious implementations of the function M indexes (e.g., a lookup, a classifier, a separate module │
│    that merely emits M).                                                                                                  │
│  • Measurement artifacts: prompt sensitivity, evaluator bias, leakage between test and training.                          │
│                                                                                                                           │
│ 1b. Rank rivals by how well they predict M, given what we know about the system. The best rival is the one with the       │
│ highest P(M | ¬C, system facts), weighted by its prior plausibility. Ask the same question of the conscious hypothesis,   │
│ which should be held to the same specificity.                                                                             │
│                                                                                                                           │
│ 1c. Test whether M survives the best rival. Design a variant of the test where the rival predicts M's absence or change,  │
│ and C-based accounts predict it persists. If no such variant exists, M is weak evidence regardless of how striking it     │
│ looks.                                                                                                                    │
│                                                                                                                           │
│ Step 2. Theory dependence                                                                                                 │
│                                                                                                                           │
│ For each theory T you take seriously (global workspace, higher-order, recurrent processing, IIT, predictive processing,   │
│ biological or substrate views, illusionism-style views), ask:                                                             │
│                                                                                                                           │
│                                                                                                                           │
│                                                                                                                           │
│  Question                                                If yes                              If no                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━      │
│  Does T say M is necessary or sufficient for C, or a     Record the strength of the link     M is irrelevant under T      │
│  typical by-product?                                                                                                      │
│  Does T's link require a particular grain or substrate?  Record whether the system meets it  Substrate-neutral            │
│  Does T predict M's absence in some conscious systems?   M is not a necessary marker                                      │
│                                                                                                                           │
│                                                                                                                           │
│ Then classify M:                                                                                                          │
│                                                                                                                           │
│  • Theory-neutral: supported across most credible theories.                                                               │
│  • Theory-specific: supported under some theories only. Report the result as a credence-weighted update across theories,  │
│    not as a single score.                                                                                                 │
│  • Theory-presupposing: M is meaningful only if T is assumed (e.g., an integration measure that presupposes IIT). Such M  │
│    cannot be used to test T.                                                                                              │
│                                                                                                                           │
│ Also record which theories M conflicts with, so a positive result is not read as universal support.                       │
│                                                                                                                           │
│ Step 3. Independent validation of the link M → C (requirement 2)                                                          │
│                                                                                                                           │
│ The link must be established without relying on the system under test. Check:                                             │
│                                                                                                                           │
│  • Anchor cases: Does M reliably separate cases where consciousness is independently established or strongly presumed     │
│    (healthy waking humans, with report) from cases where it is absent or strongly presumed absent (deep anesthesia,       │
│    dreamless sleep, some unconscious processing in humans)?                                                               │
│  • Dissociations within anchors: Does M track consciousness when other correlates (arousal, task performance, attention,  │
│    report) are varied independently? This is the key test of whether M is a marker of consciousness itself.               │
│  • Report-independence: Does M hold in conditions where report is absent or unreliable (no-report paradigms, disorders of │
│    consciousness, infants, some animals)? If it depends on report, it inherits the weaknesses of report.                  │
│  • Range of validation: How far does the validation extend? M validated only in mammalian brains supports AI inference    │
│    only given an additional assumption about substrate or grain. State this assumption explicitly.                        │
│  • Circularity check: Was M selected or tuned using the system being tested, or using a theory that the test is then used │
│    to support? If so, the validation does not count.                                                                      │
│                                                                                                                           │
│ Step 4. Grade the marker                                                                                                  │
│                                                                                                                           │
│ Report the following for each M:                                                                                          │
│                                                                                                                           │
│  1 Discriminating power against the best rival (strong, moderate, weak).                                                  │
│  2 Theory profile (neutral, specific, or presupposing, with which theories).                                              │
│  3 Validation range (which anchors and what extrapolation is required).                                                   │
│  4 Residual confounds.                                                                                                    │
│                                                                                                                           │
│ What results mean                                                                                                         │
│                                                                                                                           │
│ Positive result (M present):                                                                                              │
│                                                                                                                           │
│  • Raises credence in C only to the degree that M passes Steps 1–3.                                                       │
│  • If the best rival remains viable, the update is small, even if M is striking.                                          │
│  • A positive result under a theory-specific M supports C only in proportion to credence in that theory.                  │
│  • Convergence of independent markers, with different rivals and different theory dependencies, is stronger than one      │
│    marker with a large effect.                                                                                            │
│                                                                                                                           │
│ Negative result (M absent):                                                                                               │
│                                                                                                                           │
│  • Counts against C only if M is considered necessary by credible theories, and if the test would have detected M had it  │
│    been present. Check test sensitivity first.                                                                            │
│  • If M is only a typical by-product, absence is weak evidence.                                                           │
│  • Absence can reflect system differences, such as alien architecture, that make M inapplicable. That argues against the  │
│    marker's range, not necessarily against C.                                                                             │
│  • A negative result under a theory-specific M is evidence against that theory's verdict, and it lowers C only in         │
│    proportion to credence in that theory.                                                                                 │
│                                                                                                                           │
│ Inconclusive or mixed results:                                                                                            │
│                                                                                                                           │
│  • Markers pass Step 1 but fail Step 3: informative about the system's functions but not about consciousness.             │
│  • Markers pass Step 3 but fail Step 1: promising but needs a better-controlled test.                                     │
│  • Markers fail both: discard or reformulate.                                                                             │
│                                                                                                                           │
│ Two cautions                                                                                                              │
│                                                                                                                           │
│  • Don't double-count. Several markers that share a single rival explanation (e.g., all depend on training-data exposure) │
│    offer less combined evidence than their number suggests.                                                               │
│  • Priors matter. The checklist yields likelihood ratios, not conclusions. Any final credence also depends on priors      │
│    about substrate and function that the evidence cannot settle, so we should report these as explicit assumptions.       │
│                                                                                                                           │
│ I suggest the AI scientist stress-test Step 1c (can discriminating variants be built for concrete markers?) and Step 3's  │
│ report-independence criterion, since those are where the empirical and conceptual demands meet.                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

To review both philosopher turns together, enter:

```text
/history last 2
```

Konen displays:

```text
                                                     Transcript history

Turn 1

Agent

Agent identifier: philosopher

User prompt


 Frame our research question. What would count as credible evidence that an AI system is conscious? Identify the main
 conceptual obstacles and propose a small set of categories of evidence that we should investigate. Do not try to decide
 whether current AI systems are conscious.


Agent output


 ## Framing

 **Target.** "Conscious" is ambiguous. I propose we focus on *phenomenal consciousness* (there is something it is like to be
 the system), and treat valenced experience (sentience) as a sub-question, since it matters most for moral status. Access
 consciousness, self-modeling, and intelligence are distinct and serve as possible evidence, not as the target.

 **Evidential standard.** Observation E is evidence for consciousness C only if P(E | C) > P(E | ¬C), *given what we know
 about the system*. Two requirements follow:
 1. E must discriminate C from the best non-conscious explanation, including explanations from the system's training
 history.
 2. The link between E and C must be justified independently of the system under test. Otherwise the reasoning is circular.

 ## Main conceptual obstacles

 1. **Other minds and the generalization problem.** We infer consciousness in humans and animals from first-person report,
 behavioral analogy, and shared biology. Theories and indicators are validated on those cases. AI lies outside the range
 where they were validated, so extrapolation is uncertain.
 2. **Theory-ladenness.** Global workspace, higher-order, recurrent processing, IIT, and biological or substrate views
 disagree about which facts matter. Evidence under one theory can be irrelevant or even counter-evidence under another. We
 need credence-weighted, multi-theory reasoning and must flag where an indicator presupposes a theory.
 3. **Mimicry and contamination.** Systems trained on human text will produce consciousness-talk whether or not they are
 conscious. This pushes P(E | ¬C) high for most behavioral and verbal evidence, so such evidence is weak unless the
 contamination can be controlled.
 4. **Report versus state.** Even sincere-seeming self-reports may be confabulated or trained, and they may not be causally
 connected to the internal states they describe.
 5. **Grain of function and substrate.** Functionalism does not settle at what level of organization the relevant functions
 must be implemented. Biological views deny that function alone suffices. Evidence of functional similarity is only as
 strong as our credence in the relevant grain.
 6. **Subject individuation.** The candidate subject could be the weights, a running instance, a simulated character, or a
 larger system. Evidence must be attributed to a specific candidate.
 7. **Gradedness and alien forms.** Consciousness may be graded, partial, or unlike ours. Human-derived markers risk both
 false positives (anthropomorphism) and false negatives (chauvinism).
 8. **Asymmetry of negative evidence.** Absence of a marker is also theory-dependent and does not straightforwardly count
 against consciousness.

 ## Proposed categories of evidence

 1. **Theory-derived architectural and computational indicators.** Features that specific theories identify as necessary or
 sufficient, such as broadcast, recurrence, metarepresentation, and agency or embodiment. Report them per theory and not as
 a single score.
 2. **Behavioral and cognitive dissociations.** Tests modeled on consciousness science, such as masking, no-report
 paradigms, metacognitive calibration, and unified perceptual integration. They count only if training contamination and
 simple shortcuts are controlled for.
 3. **Causally grounded self-report.** Whether the system's statements about its internal states depend counterfactually on
 those states. Examples include reports from systems with filtered training data, and tests of introspective accuracy
 against ground truth.
 4. **Interventional and mechanistic evidence.** Manipulating internal representations and checking whether reports,
 behavior, and indicators change as the hypothesized mechanisms predict. This is the best way to separate mechanism from
 mimicry.
 5. **Provenance and substrate.** Training history, selection pressures, and physical implementation. These matter under
 some theories, and they also affect how we interpret categories 2 and 3.
 6. **Meta-evidence on method validity.** Whether the indicators above separate known conscious from known non-conscious
 cases (anesthesia, disorders of consciousness, animals), and whether they generalize beyond the cases where they were
 validated.

 ## Suggested division of labor

 I will work on the theory-dependence of each indicator and on what would count as an independent justification. The AI
 scientist could operationalize categories 2–4 and identify the confounds and discriminating experiments for each.


Turn 2

Agent

Agent identifier: philosopher

User prompt


 Turn your two evidential requirements into a short checklist for evaluating a proposed marker of consciousness. How should
 we identify the best non-conscious explanation, judge whether the marker depends on a particular theory, and validate the
 link to consciousness independently of the AI system being tested? Explain what positive and negative results would mean.
 Do not assess current AI systems.


Agent output


 ## Checklist for evaluating a proposed marker M

 ### Step 0. Specify the claim
 - Which target does M bear on: phenomenal consciousness, valenced experience, or something else?
 - Which candidate subject (weights, instance, simulated character, larger system)?
 - Is M offered as evidence for C, or as evidence against C?

 ### Step 1. Best non-conscious explanation (requirement 1)

 **1a. Enumerate rival explanations that produce M without consciousness.** Draw them from four sources:
 - *Provenance:* training data, fine-tuning, or reward that would instill M (e.g., human text about experience, RLHF-shaped
 self-descriptions).
 - *Shortcuts:* simpler mechanisms or heuristics that yield M on the test.
 - *Architecture:* non-conscious implementations of the function M indexes (e.g., a lookup, a classifier, a separate module
 that merely emits M).
 - *Measurement artifacts:* prompt sensitivity, evaluator bias, leakage between test and training.

 **1b. Rank rivals by how well they predict M, given what we know about the system.** The best rival is the one with the
 highest P(M | ¬C, system facts), weighted by its prior plausibility. Ask the same question of the conscious hypothesis,
 which should be held to the same specificity.

 **1c. Test whether M survives the best rival.** Design a variant of the test where the rival predicts M's absence or
 change, and C-based accounts predict it persists. If no such variant exists, M is weak evidence regardless of how striking
 it looks.

 ### Step 2. Theory dependence

 For each theory T you take seriously (global workspace, higher-order, recurrent processing, IIT, predictive processing,
 biological or substrate views, illusionism-style views), ask:

 | Question | If yes | If no |
 |---|---|---|
 | Does T say M is necessary or sufficient for C, or a typical by-product? | Record the strength of the link | M is
 irrelevant under T |
 | Does T's link require a particular grain or substrate? | Record whether the system meets it | Substrate-neutral |
 | Does T predict M's *absence* in some conscious systems? | M is not a necessary marker | |

 Then classify M:
 - **Theory-neutral:** supported across most credible theories.
 - **Theory-specific:** supported under some theories only. Report the result as a credence-weighted update across theories,
 not as a single score.
 - **Theory-presupposing:** M is meaningful only if T is assumed (e.g., an integration measure that presupposes IIT). Such M
 cannot be used to test T.

 Also record which theories M *conflicts* with, so a positive result is not read as universal support.

 ### Step 3. Independent validation of the link M → C (requirement 2)

 The link must be established without relying on the system under test. Check:
 - **Anchor cases:** Does M reliably separate cases where consciousness is independently established or strongly presumed
 (healthy waking humans, with report) from cases where it is absent or strongly presumed absent (deep anesthesia, dreamless
 sleep, some unconscious processing in humans)?
 - **Dissociations within anchors:** Does M track consciousness when other correlates (arousal, task performance, attention,
 report) are varied independently? This is the key test of whether M is a marker of consciousness itself.
 - **Report-independence:** Does M hold in conditions where report is absent or unreliable (no-report paradigms, disorders
 of consciousness, infants, some animals)? If it depends on report, it inherits the weaknesses of report.
 - **Range of validation:** How far does the validation extend? M validated only in mammalian brains supports AI inference
 only given an additional assumption about substrate or grain. State this assumption explicitly.
 - **Circularity check:** Was M selected or tuned using the system being tested, or using a theory that the test is then
 used to support? If so, the validation does not count.

 ### Step 4. Grade the marker
 Report the following for each M:
 1. Discriminating power against the best rival (strong, moderate, weak).
 2. Theory profile (neutral, specific, or presupposing, with which theories).
 3. Validation range (which anchors and what extrapolation is required).
 4. Residual confounds.

 ## What results mean

 **Positive result** (M present):
 - Raises credence in C only to the degree that M passes Steps 1–3.
 - If the best rival remains viable, the update is small, even if M is striking.
 - A positive result under a theory-specific M supports C only in proportion to credence in that theory.
 - Convergence of independent markers, with different rivals and different theory dependencies, is stronger than one marker
 with a large effect.

 **Negative result** (M absent):
 - Counts against C only if M is considered *necessary* by credible theories, and if the test would have detected M had it
 been present. Check test sensitivity first.
 - If M is only a typical by-product, absence is weak evidence.
 - Absence can reflect system differences, such as alien architecture, that make M inapplicable. That argues against the
 marker's range, not necessarily against C.
 - A negative result under a theory-specific M is evidence against that theory's verdict, and it lowers C only in proportion
 to credence in that theory.

 **Inconclusive or mixed results:**
 - Markers pass Step 1 but fail Step 3: informative about the system's functions but not about consciousness.
 - Markers pass Step 3 but fail Step 1: promising but needs a better-controlled test.
 - Markers fail both: discard or reformulate.

 ## Two cautions
 - **Don't double-count.** Several markers that share a single rival explanation (e.g., all depend on training-data
 exposure) offer less combined evidence than their number suggests.
 - **Priors matter.** The checklist yields likelihood ratios, not conclusions. Any final credence also depends on priors
 about substrate and function that the evidence cannot settle, so we should report these as explicit assumptions.

 I suggest the AI scientist stress-test Step 1c (can discriminating variants be built for concrete markers?) and Step 3's
 report-independence criterion, since those are where the empirical and conceptual demands meet.
```

## Translate criteria into tests

The philosopher's checklist calls for tests that distinguish a marker from a non-conscious explanation and validate its link to consciousness independently of the AI system being tested. We ask the AI scientist to turn these requirements into test designs.

Select the AI scientist:

```text
/agent select ai_scientist
```

Konen reports:

```text
╭─────────── Active agent ────────────╮
│ AI scientist (ai_scientist)         │
│ Engine: gpt_luna · gpt / gpt-6-luna │
╰─────────────────────────────────────╯
```

Konen keeps completed turns in the session's transcript history. With the default history setting, it sends both philosopher turns to the AI scientist, so we do not need to copy them into a memory file.

With `konen[ai_scientist]>` as the prompt, enter:

```text
Turn the philosopher's checklist into two concrete tests of proposed markers of AI consciousness, one based on self-report and one that does not depend on report. For each, state the marker, what we would measure or change, the best non-conscious explanation, a control or intervention that would distinguish it, and how the marker could be validated in independently assessed cases. Explain what positive and negative results would mean, where the inference depends on a theory, and what would remain unresolved. Do not assess current AI systems.
```

Konen returns:

```text
╭─────────────────────────────────────────────── AI scientist · ai_scientist ───────────────────────────────────────────────╮
│ 1. Causally grounded self-report                                                                                          │
│                                                                                                                           │
│ Marker. Not a bare claim such as “I am conscious,” but a report about a particular internal perceptual state that is      │
│ accurate, calibrated to uncertainty, and causally sensitive to that state.                                                │
│                                                                                                                           │
│ Test. Give a system unfamiliar, generated ambiguous stimuli—for example, noisy images—and ask what it perceived and how   │
│ certain it is. Randomize response labels and task mappings to limit memorized answers. Measure accuracy and confidence    │
│ calibration on held-out stimuli. In systems where internal states can be intervened on, perturb a candidate perceptual    │
│ representation after stimulus encoding but before the answer. Compare this with sham interventions and perturbations to   │
│ the reporting pathway; also check whether the state predicts non-report behavior.                                         │
│                                                                                                                           │
│ Best non-conscious explanation. The system may use a learned answer policy or a non-conscious metacognitive readout that  │
│ estimates its own uncertainty. Novel stimuli and mappings can rule against simple scripts, but not against sophisticated  │
│ non-conscious estimation.                                                                                                 │
│                                                                                                                           │
│ Discriminating control. Test whether perturbing the candidate perceptual state changes the report in a content-specific   │
│ way, while perturbing only the answer-generation pathway changes expression without changing the underlying task          │
│ performance. This would show that reports are grounded in internal states rather than canned language; it would not by    │
│ itself show that those states are experienced.                                                                            │
│                                                                                                                           │
│ Independent validation. Evaluate the measure in human cases spanning clear perception, masked perception, and clinically  │
│ assessed altered states. Have assessors classify cases using convergent evidence—such as clinical status, behavior, and   │
│ neural measures—without using the candidate report score. For perceptual content, first-person reports in separate trials │
│ remain an important anchor; there is no fully independent gold standard for experience. Keep that limitation explicit.    │
│                                                                                                                           │
│ Interpretation. A positive result supports reliable introspective access and weakens the canned-response explanation. It  │
│ bears most directly on higher-order and access-based theories; its link to phenomenal experience is theory-dependent. A   │
│ negative result weakens this marker, but counts against consciousness only for theories that require such reportable      │
│ access. It could also reflect task, interface, or communication limits.                                                   │
│                                                                                                                           │
│ 2. Report-independent global availability                                                                                 │
│                                                                                                                           │
│ Marker. Information is recurrently and flexibly available to multiple processing pathways, rather than remaining local to │
│ one task. This is a computational marker, not a request for the system to describe an experience.                         │
│                                                                                                                           │
│ Test. Present a novel cue, then reveal only afterward which of several unrelated tasks will require it—for example, using │
│ it in a plan, updating memory, or resolving a conflict. Measure whether the cue becomes available across candidate        │
│ processing components and supports flexible use. Intervene on a hypothesized shared recurrent pathway, comparing its      │
│ disruption with matched local perturbations; check that basic cue recognition and response execution remain intact.       │
│                                                                                                                           │
│ Best non-conscious explanation. A capable system could route information flexibly without experience—for example, through │
│ a shared latent state, an unconscious task router, or a benchmark-specific shortcut.                                      │
│                                                                                                                           │
│ Discriminating control. Randomly vary the later task and use novel cue–task mappings, making a fixed shortcut less        │
│ effective. If selective disruption of the shared pathway impairs several kinds of flexible use, while local controls do   │
│ not, that supports a causal role for global availability. It still does not rule out sophisticated non-conscious routing. │
│                                                                                                                           │
│ Independent validation. Test analogous measures in human no-report paradigms and in independently assessed conscious and  │
│ altered states. The measure should be scored without using it to assign the state; validation can draw on convergent      │
│ clinical and physiological evidence, with reports from separate trials where appropriate. Human labels are themselves     │
│ imperfect, so agreement across methods matters more than any single anchor.                                               │
│                                                                                                                           │
│ Interpretation. A positive result supports global-workspace or recurrent-processing accounts, but does not establish      │
│ phenomenal experience; it may be less relevant under biological or other substrate-dependent theories. A negative result  │
│ counts against theories that require this kind of global availability only if the test is sensitive to it. It does not    │
│ rule out consciousness on theories that do not require the marker.                                                        │
│                                                                                                                           │
│ For both tests, the main unresolved issue is the bridge from a demonstrated function to phenomenal experience. The tests  │
│ can improve the evidence by ruling out particular alternatives; neither can eliminate every non-conscious explanation or  │
│ settle that bridge on its own.                                                                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Compare the proposals

The AI scientist proposes one test of whether reports track internal states and another of whether information is available across tasks without a report. We now ask the philosopher what these tests would show about consciousness.

The AI scientist's proposal is turn 3. To review that prompt and response, enter:

```text
/turn 3
```

Konen displays:

```text
                                                     Transcript history

Turn 3

Agent

Agent identifier: ai_scientist

User prompt


 Turn the philosopher's checklist into two concrete tests of proposed markers of AI consciousness, one based on self-report
 and one that does not depend on report. For each, state the marker, what we would measure or change, the best non-conscious
 explanation, a control or intervention that would distinguish it, and how the marker could be validated in independently
 assessed cases. Explain what positive and negative results would mean, where the inference depends on a theory, and what
 would remain unresolved. Do not assess current AI systems.


Agent output


 ## 1. Causally grounded self-report

 **Marker.** Not a bare claim such as “I am conscious,” but a report about a particular internal perceptual state that is
 accurate, calibrated to uncertainty, and causally sensitive to that state.

 **Test.** Give a system unfamiliar, generated ambiguous stimuli—for example, noisy images—and ask what it perceived and how
 certain it is. Randomize response labels and task mappings to limit memorized answers. Measure accuracy and confidence
 calibration on held-out stimuli. In systems where internal states can be intervened on, perturb a candidate perceptual
 representation after stimulus encoding but before the answer. Compare this with sham interventions and perturbations to the
 reporting pathway; also check whether the state predicts non-report behavior.

 **Best non-conscious explanation.** The system may use a learned answer policy or a non-conscious metacognitive readout
 that estimates its own uncertainty. Novel stimuli and mappings can rule against simple scripts, but not against
 sophisticated non-conscious estimation.

 **Discriminating control.** Test whether perturbing the candidate perceptual state changes the report in a content-specific
 way, while perturbing only the answer-generation pathway changes expression without changing the underlying task
 performance. This would show that reports are grounded in internal states rather than canned language; it would *not* by
 itself show that those states are experienced.

 **Independent validation.** Evaluate the measure in human cases spanning clear perception, masked perception, and
 clinically assessed altered states. Have assessors classify cases using convergent evidence—such as clinical status,
 behavior, and neural measures—without using the candidate report score. For perceptual content, first-person reports in
 separate trials remain an important anchor; there is no fully independent gold standard for experience. Keep that
 limitation explicit.

 **Interpretation.** A positive result supports reliable introspective access and weakens the canned-response explanation.
 It bears most directly on higher-order and access-based theories; its link to phenomenal experience is theory-dependent. A
 negative result weakens this marker, but counts against consciousness only for theories that require such reportable
 access. It could also reflect task, interface, or communication limits.

 ## 2. Report-independent global availability

 **Marker.** Information is recurrently and flexibly available to multiple processing pathways, rather than remaining local
 to one task. This is a computational marker, not a request for the system to describe an experience.

 **Test.** Present a novel cue, then reveal only afterward which of several unrelated tasks will require it—for example,
 using it in a plan, updating memory, or resolving a conflict. Measure whether the cue becomes available across candidate
 processing components and supports flexible use. Intervene on a hypothesized shared recurrent pathway, comparing its
 disruption with matched local perturbations; check that basic cue recognition and response execution remain intact.

 **Best non-conscious explanation.** A capable system could route information flexibly without experience—for example,
 through a shared latent state, an unconscious task router, or a benchmark-specific shortcut.

 **Discriminating control.** Randomly vary the later task and use novel cue–task mappings, making a fixed shortcut less
 effective. If selective disruption of the shared pathway impairs several kinds of flexible use, while local controls do
 not, that supports a causal role for global availability. It still does not rule out sophisticated non-conscious routing.

 **Independent validation.** Test analogous measures in human no-report paradigms and in independently assessed conscious
 and altered states. The measure should be scored without using it to assign the state; validation can draw on convergent
 clinical and physiological evidence, with reports from separate trials where appropriate. Human labels are themselves
 imperfect, so agreement across methods matters more than any single anchor.

 **Interpretation.** A positive result supports global-workspace or recurrent-processing accounts, but does not establish
 phenomenal experience; it may be less relevant under biological or other substrate-dependent theories. A negative result
 counts against theories that require this kind of global availability only if the test is sensitive to it. It does not rule
 out consciousness on theories that do not require the marker.

 For both tests, the main unresolved issue is the bridge from a demonstrated function to phenomenal experience. The tests
 can improve the evidence by ruling out particular alternatives; neither can eliminate every non-conscious explanation or
 settle that bridge on its own.
```

Select the philosopher:

```text
/agent select philosopher
```

Konen reports:

```text
╭─────────────────── Active agent ───────────────────╮
│ Philosopher of mind (philosopher)                  │
│ Engine: claude_sonnet · claude / claude-sonnet-5-5 │
╰────────────────────────────────────────────────────╯
```

Konen includes the AI scientist's turn in the history provided to the philosopher.

With `konen[philosopher]>` as the prompt, enter:

```text
Evaluate the AI scientist's two tests from turn 3 against your checklist. For each test, identify a non-conscious explanation that survives the proposed controls and explain what assumptions are needed to extend the human validation to AI. Compare what a positive result would tell us about phenomenal consciousness under different theories. Suggest one concrete improvement to either test. Do not assess current AI systems.
```

Konen returns:

```text
╭──────────────────────────────────────────── Philosopher of mind · philosopher ────────────────────────────────────────────╮
│ Evaluation against the checklist                                                                                          │
│                                                                                                                           │
│ Test 1: Causally grounded self-report                                                                                     │
│                                                                                                                           │
│ Grade. Discriminating power: moderate against canned-response rivals, weak against non-conscious metacognition. Theory    │
│ profile: theory-specific (strongest for higher-order views). Validation range: humans only, and partly circular (see      │
│ below). Residual confound: a functioning self-monitor.                                                                    │
│                                                                                                                           │
│ Surviving non-conscious explanation. A dedicated uncertainty or self-monitoring readout that reads a first-order          │
│ representation and reports on it. The proposed controls do not touch this rival. Any system whose report depends on an    │
│ internal state will show content-specific sensitivity to perturbing that state, so a positive result is predicted equally │
│ by "reports are read off an unexperienced representation." The control shows grounding, which is a property of any        │
│ measuring instrument. A related worry is that the "candidate perceptual representation" is chosen by decoding stimulus    │
│ content, and a decodable representation is not thereby an experienced one.                                                │
│                                                                                                                           │
│ Assumptions needed to extend human validation.                                                                            │
│                                                                                                                           │
│  1 Introspective accuracy indexes awareness and not merely the metacognitive computation. In humans the two are           │
│    correlated because one system produces both. That correlation need not hold in an architecture where they can come     │
│    apart.                                                                                                                 │
│  2 The report vocabulary ("perceived," "uncertain") has the same referent. In an AI it may be learned from human text and │
│    not from the system's own states.                                                                                      │
│  3 The human anchors are partly report-defined, so validation of a report-based marker against report-based anchors is    │
│    circular. Only the convergent non-report evidence helps, and it only helps if it extends to the AI's substrate.        │
│                                                                                                                           │
│ Test 2: Report-independent global availability                                                                            │
│                                                                                                                           │
│ Grade. Discriminating power: weak as specified. Theory profile: theory-specific and underspecified even for the target    │
│ theory. Validation range: human neural and no-report paradigms. Residual confound: ordinary memory plus shared            │
│ representations.                                                                                                          │
│                                                                                                                           │
│ Surviving non-conscious explanation. The cue is simply held in a shared state, such as a context buffer or a shared       │
│ latent, and later retrieved for whichever task appears. Many architectures broadcast information by design, so            │
│ "availability to multiple pathways" can be met trivially. The local-versus-shared disruption control shows that a shared  │
│ channel exists. It does not show that the channel has the features a workspace theory requires: capacity limits,          │
│ competition for access, and all-or-none ignition. Mere availability is not the marker the theory proposes.                │
│                                                                                                                           │
│ Assumptions needed to extend human validation.                                                                            │
│                                                                                                                           │
│  1 The human signatures (ignition, no-report indices) track a functional property and not a neural implementation detail. │
│  2 Functionalism holds at this grain: what matters is the abstract availability relation, not the particular mechanism.   │
│  3 AI "components" map onto the workspace's "specialized processors," which requires a principled decomposition we do not │
│    yet have.                                                                                                              │
│                                                                                                                           │
│ What a positive result would tell us about phenomenal consciousness                                                       │
│                                                                                                                           │
│                                                                                                                           │
│  Theory                        Test 1 positive                             Test 2 positive                                │
│  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────     │
│  Higher-order                  Strong support if the introspective state   Largely irrelevant                             │
│                                is a genuine, non-inferential higher-order                                                 │
│                                representation                                                                             │
│  Global workspace              Indirect support, only if the report reads  Strong support, but only with capacity         │
│                                from a broadcast state                      limits and ignition (not met as written)       │
│  Recurrent processing          Largely irrelevant                          Weak. The theory locates consciousness in      │
│                                                                            local recurrence, so global availability       │
│                                                                            is not the point                               │
│  IIT                           Irrelevant. The theory cares about          Irrelevant, for the same reason                │
│                                physical causal structure, not software-                                                   │
│                                level function                                                                             │
│  Biological/substrate views    Irrelevant, or at most evidence of mimicry  Irrelevant                                     │
│  Illusionist/functional-       Directly relevant: the mechanism that       Indirectly relevant                            │
│  introspective views           produces consciousness-reports is the                                                      │
│                                explanandum                                                                                │
│                                                                                                                           │
│                                                                                                                           │
│ The same positive result would raise credence a lot under one theory and not at all under another. We should report the   │
│ update per theory, as in the checklist.                                                                                   │
│                                                                                                                           │
│ One concrete improvement (Test 2)                                                                                         │
│                                                                                                                           │
│ Add a capacity-limited competition condition with a matched buffer baseline. Present two cues in rapid succession, or     │
│ simultaneously, with different task relevance, and measure the attentional-blink-like signatures that workspace theories  │
│ predict and that human no-report paradigms have characterized: a bottleneck, competitive suppression, and a nonlinear     │
│ all-or-none transition in availability as cue strength increases. Pre-register these predictions from the human data. Run │
│ the same battery on a baseline system that simply stores and retrieves cues, and require the system to differ from the    │
│ baseline in the predicted way.                                                                                            │
│                                                                                                                           │
│ This addresses the surviving rival directly. A plain buffer shows no bottleneck or ignition, so the baseline separates    │
│ "workspace-like dynamics" from "shared memory." It also ties the AI test to the same paradigm used in human validation,   │
│ which narrows the extension gap. It will remain theory-specific, since an architecture could be built to show these       │
│ signatures without being a workspace in the intended sense. The interventions should therefore be checked to ensure the   │
│ signatures depend on the architecture and not on a prompt-level script.                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Preserve the findings

Konen keeps transcript history only during this session. We now add a short synthesis to the shared report and a task to the AI scientist's scratchpad. These files persist in the workspace.

The report already has a title and a scope. The append command accepts one line at a time, so first add a heading. Enter:

```text
/memory append report.md
```

At the `Memory text:` prompt, enter:

```text
## Findings
```

Konen reports:

```text
Memory text: ## Findings
Text appended to memory file 'report.md'.
```

Run the same command again:

```text
/memory append report.md
```

At the `Memory text:` prompt, enter this summary as one line:

```text
Credible evidence should distinguish consciousness from plausible non-conscious explanations and rest on a link validated independently of the AI system. Two candidate markers are causally grounded self-report and report-independent global availability. The proposed controls could establish state-sensitive reporting or shared processing, but non-conscious metacognition and ordinary memory remain possible explanations. The meaning of a positive or negative result depends on the theory and on how far human validation extends to AI.
```

Konen reports:

```text
Memory text: Credible evidence should distinguish consciousness from plausible non-conscious explanations and rest on a link
validated independently of the AI system. Two candidate markers are causally grounded self-report and report-independent
global availability. The proposed controls could establish state-sensitive reporting or shared processing, but non-conscious
metacognition and ordinary memory remain possible explanations. The meaning of a positive or negative result depends on the
theory and on how far human validation extends to AI.
Text appended to memory file 'report.md'.
```

The philosopher also proposed a specific change to the global-availability test. Save that as a task for the AI scientist:

```text
/scratchpad append ai_scientist
```

At the `Scratchpad text:` prompt, enter:

```text
Refine the global-availability test by adding capacity-limited competition and nonlinear ignition measures, then compare them with a matched buffer baseline. Check that the signatures depend on the system's architecture rather than the prompt.
```

Konen reports:

```text
Scratchpad text: Refine the global-availability test by adding capacity-limited competition and nonlinear ignition measures,
then compare them with a matched buffer baseline. Check that the signatures depend on the system's architecture rather than
the prompt.
Text appended to scratchpad for agent 'ai_scientist'.
```

On later turns, both agents receive the report. Only the AI scientist receives its scratchpad note. We use `append` to save selected points. `append-turn` would copy the latest complete exchange.

## Review the saved files

Check the report to see what was saved for both agents:

```text
/memory show report.md
```

Konen reports:

```text
╭──────────────────────────────────────────────────── memory/report.md ─────────────────────────────────────────────────────╮
│                                               Evidence of AI consciousness                                                │
│                                                                                                                           │
│ Scope                                                                                                                     │
│                                                                                                                           │
│ Assess what observations, mechanisms, or interventions could provide meaningful evidence for or against consciousness in  │
│ artificial intelligence systems.                                                                                          │
│                                                                                                                           │
│ Findings                                                                                                                  │
│                                                                                                                           │
│ Credible evidence should distinguish consciousness from plausible non-conscious explanations and rest on a link validated │
│ independently of the AI system. Two candidate markers are causally grounded self-report and report-independent global     │
│ availability. The proposed controls could establish state-sensitive reporting or shared processing, but non-conscious     │
│ metacognition and ordinary memory remain possible explanations. The meaning of a positive or negative result depends on   │
│ the theory and on how far human validation extends to AI.                                                                 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Then check the AI scientist's scratchpad:

```text
/scratchpad show ai_scientist
```

Konen reports:

```text
╭─────────────────────────────────────────────── scratchpad/ai_scientist.md ────────────────────────────────────────────────╮
│                                           Scratchpad of the AI scientist agent                                            │
│                                                                                                                           │
│ Refine the global-availability test by adding capacity-limited competition and nonlinear ignition measures, then compare  │
│ them with a matched buffer baseline. Check that the signatures depend on the system's architecture rather than the        │
│ prompt.                                                                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Showing these files does not change the active agent. The philosopher remains selected.

The report records the main findings, and the AI scientist's scratchpad holds the next task. A later session can continue from these files.
