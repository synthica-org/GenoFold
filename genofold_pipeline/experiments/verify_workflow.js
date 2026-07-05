export const meta = {
  name: 'genofold-verify',
  description: 'Adversarially verify GenoFold paper claims against the experiment code and result JSONs',
  phases: [
    { title: 'Verify' },
    { title: 'Synthesize' },
  ],
}

// Independent verification dimensions. Each agent is a skeptic tasked to find a
// real problem; "looks fine" is only acceptable after a genuine check.
const REPO = '/Users/buno/Documents/coding/synthica-org/GenoFold'
const PIPE = `${REPO}/genofold_pipeline`

const DIMS = [
  {
    key: 'metric-correctness',
    prompt: `You are auditing a computational-biology paper's METRIC code for correctness.
Read ${PIPE}/experiments/metrics.py. Verify that:
- dSCC (distance Spearman correlation) is computed correctly: Spearman between reconstructed
  pairwise distances and contact-derived "wish" distances over observed pairs, and that a
  perfect reconstruction would score ~1 and a random one ~0 (the module's __main__ self-check asserts this).
- wish_distance, observed_pair_mask, distance_matching_loss, calibration, and tad_boundary_f1 are
  mathematically sound and not accidentally trivial or leaking.
Report any real bug with file:line and a concrete failure scenario. If the metric is sound, say so explicitly with evidence.`,
  },
  {
    key: 'benchmark-fairness',
    prompt: `You are auditing whether the GenoFold benchmark is a FAIR head-to-head.
Read ${PIPE}/experiments/baselines.py, ${PIPE}/experiments/run_experiments.py (run_main_benchmark),
and ${PIPE}/experiments/train.py. Check:
- Do GenoFold and all baselines (ShRec3D, 3DMax, HiC-GNN, FLAMINGO-style) consume the SAME
  valid-bin contact submatrix and get scored by the SAME dSCC over the same pairs?
- Is GenoFold's main-benchmark number computed on the same data regime (all observed pairs) as the baselines,
  rather than a handicapped held-out set?
- Are any baselines crippled (too few epochs/restarts) relative to GenoFold in a way that inflates GenoFold?
Report concrete unfairness with file:line, or confirm fairness with evidence.`,
  },
  {
    key: 'claims-vs-data',
    prompt: `You are fact-checking a paper's claims against its actual results.
Read the paper ${REPO}/paper/GenoFold_BIBM26.tex and the result JSONs in ${PIPE}/results/*.json plus
${REPO}/paper/tables/macros.tex. For each quantitative claim in the abstract/results (dSCC "on par",
generalization train vs held-out, cross-cell transfer, ablation gain in sparse regions, calibration Spearman,
scaling), check it is supported by the JSON numbers and NOT overstated. Flag any claim the data does not
support (e.g. claiming GenoFold "beats" baselines if it only matches; claiming "well-calibrated" if the
Spearman is weak; claiming FISH validation which is not done). Quote the JSON value next to each claim.`,
  },
  {
    key: 'data-realness',
    prompt: `You are verifying the paper's central rebuttal: that GenoFold now uses REAL data and a REAL trained model.
Read ${PIPE}/experiments/data_fetch.py, ${PIPE}/experiments/prep.py, and ${PIPE}/experiments/train.py. Confirm:
- Hi-C is streamed from real public .hic files (Rao 2014), not synthetic.
- Mappability is a real ENCODE/CRG track (not the old 0.9 placeholder), and GC/sequence come from the real hg19 reference.
- The network is actually trained (a real optimization loop with a real loss), not random-weights inference.
Flag any place where a placeholder or synthetic value silently remains. Confirm with file:line evidence.`,
  },
]

phase('Verify')
const findings = await parallel(DIMS.map(d => () =>
  agent(d.prompt, { label: `verify:${d.key}`, phase: 'Verify', schema: {
    type: 'object',
    properties: {
      dimension: { type: 'string' },
      issues: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'none'] },
            summary: { type: 'string' },
            location: { type: 'string' },
            evidence: { type: 'string' },
          },
          required: ['severity', 'summary'],
        },
      },
      verdict: { type: 'string' },
    },
    required: ['dimension', 'issues', 'verdict'],
  } })
)).then(rs => rs.filter(Boolean))

phase('Synthesize')
const summary = await agent(
  `You are the lead reviewer. Here are four independent verification reports on the GenoFold paper and code:\n\n` +
  JSON.stringify(findings, null, 2) +
  `\n\nProduce a concise verdict: is the paper's core claim (trained, benchmarked, real-data, generalizing) SUPPORTED and free of overclaims? ` +
  `List any blocker/major issues that must be fixed before submission, each with its location and the concrete fix. ` +
  `If there are none, say so plainly.`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return { findings, summary }
