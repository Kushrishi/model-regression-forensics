# Model Regression Forensics roadmap

Updated 10 October 2026. [Current evidence](STATE.md) · [Start with saved predictions](../docs/import-predictions.md)

## Deliverable

A local classification-release investigation application. An engineer supplies
baseline, candidate and repair predictions, inspects affected examples, checks
protected behavior and new-capability requirements, and keeps a portable decision
record. Successful repair and unique historical cause remain different claims.

The product goal is useful release investigation. A research paper is conditional
on a demonstrated contribution beyond ordinary comparison and rollback.

## Milestones and decisions

| Milestone | Status | Work and completion evidence | Decision afterward |
| --- | --- | --- | --- |
| Executable comparison and repair inspection | Complete on documented examples | Saved-prediction import, explicit execution, identical-case validation, slice requirements and portable HTML. Digit, Banking77 and incremental-intent examples exercise actual models. | Keep these as integration evidence; their constructed faults do not establish diagnostic advantage. |
| Deployment conversion inspection | Complete on one retained linear model | [Float32/INT8 comparison](../docs/deployment-comparison.md): exact source recovery, fixed development policy, eight changed INT8 labels, measured single-host sizes/latency and runnable saved predictions. | Prefer the label-preserving export unless the measured size/latency tradeoff warrants changed behavior; this is not independent-user or comparative research evidence. |
| Recover and package a reproducible investigation | Partly complete | Existing Banking77 bundle is retained. Recover the complete incremental-intent archive or mark it unavailable; verify hashes and rebuild reports without fitting. Document supported installation, inputs, outputs and license terms. | Do not advertise a complete downloadable experiment while its full model/input archive is unavailable. |
| Independent first use | Not completed | An engineer uses their own saved predictions through the documented importer, inspects a failed slice and exports the cases supporting a release/repair decision. Record setup failures, interpretation errors and missing capabilities. | Fix observed obstacles. One successful session establishes usability on that case, not research efficacy or broad time savings. |
| Comparative investigation evaluation | Design pending | Qualify incident provenance and declare tasks, information access, success criteria and budget before scoring. Compare with full-diff inspection, direct prediction comparison, complete rollback, targeted rollback and full retraining where applicable. | Test whether MRF changes decision quality, protected-behavior failures, unjustified unique-cause claims or investigation cost. Include ordinary methods that win. |
| Focused package and research decision | Conditional | Reproducible installation and first use, retained examples, limitations and explicit license. A study must report incident-level results, execution costs, abstentions and failures. | Release useful software independently of paper novelty. Pursue a paper only if comparative evidence supports a distinct contribution. |

## Immediate order

1. Use the complete [deployment comparison](../docs/deployment-comparison.md) on
   a second machine: import saved predictions, open the report, inspect failed
   requirements and export the cases supporting a release decision. No training
   is needed. Maintainer use is not an independent-user evaluation.
2. Prepare independent first use using the existing importer. The report must
   distinguish failed policy requirements from import errors. Keep the older
   experiment-archive gap explicit; it does not block the complete deployment
   example or use of an engineer's own saved predictions.
3. Use concrete user problems to select product changes. Do not add another model
   or dashboard feature solely to increase project scope.
4. Qualify real incidents for a prospective comparative design. Existing screens
   found no eligible training-release incident; repeated broad searches without new
   evidence are not the next experiment.
5. Decide whether the next deliverable is an improved investigation package or a
   research study from those observations.

## Study requirements before new comparative claims

Each incident needs identifiable known-good and regressed versions, complete
visible changes, supported data access and repeated task-outcome evidence. The
comparison must give ordinary baselines the same information and account for
retraining and intervention costs. Repeated seeds and overlapping constructed
scenarios are not independent incidents. A passing repair is not a unique-cause
label; multiple successful repairs remain a valid outcome.

A user evaluation and a new trained-model study need their own prospective design.
This roadmap does not reopen the failed small-model experiment or authorize
another training campaign. The official Banking77 test split remains unopened.

## What would change the direction?

- If users obtain the same decisions more simply with direct comparisons, reduce
  the package to the parts that remove demonstrated friction.
- If rollback or full retraining is the best repair, recommend it. The tool's
  purpose is a defensible decision, not making its own method win.
- If qualified incidents remain unavailable, finish a bounded engineering artifact
  and the existing negative technical report instead of manufacturing a benchmark.
- If selective diagnosis adds value, test both incorrect specific claims and the
  cost of abstaining; refusing every diagnosis is not a successful method.

## Relevant standards

Reviewed 10 October 2026:

- [mlwhatif](https://github.com/stefan-grafberger/mlwhatif) executes changes to
  native ML pipelines. Intervention execution is established work, so a callback
  runner alone is not a research contribution.
- [dattri](https://github.com/TRAIS-Lab/dattri) provides attribution methods and
  benchmarks. Appropriate attribution comparisons need matched information and cost;
  attribution rank alone does not establish successful repair or unique cause.
- [FiftyOne](https://github.com/voxel51/fiftyone) connects model evaluation with
  inspection of concrete examples. Adopt that connection while keeping MRF focused
  on local classification-release decisions rather than rebuilding a dataset platform.

## Maintained scope

Exact-label classification is the supported task. Imported execution costs are
unknown unless independently measured. LLM evaluation, causal certification,
probability calibration and general automatic repair are not current capabilities.
The completed localization study and unsuccessful development attempts remain
public evidence. Distribution licensing is unresolved; public source visibility
must not be presented as an open-source license or a published package release.
