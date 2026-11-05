"""Persistent, object-centric evidence container.

The AgentRunner previously kept only ``latest_structured`` — the most recent
tool result.  The store replaces that with an accumulating, object-centric state:
observations, claims, object hypotheses, candidate edges, relations, provenance,
coverage, and the last inference result, all surviving across rounds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from DS_VI.models.shared_identity import adapters as adapter_pkg
from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.observation_builder import (
    object_hypotheses_from_observations,
)
from DS_VI.models.shared_identity.provenance import (
    ProvenanceGraph,
    compute_evidence_fingerprint,
)
from DS_VI.models.shared_identity.schema import (
    Claim,
    InferenceProblem,
    InferenceState,
    ObjectHypothesis,
    PrimitiveObservation,
    Relation,
    SpatialSupport,
)


class EvidenceStore:
    """Accumulates evidence across tool calls and runs joint inference on demand."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config: Dict[str, Any] = dict(config or load_config())
        self.observations: Dict[str, PrimitiveObservation] = {}
        self.claims: Dict[str, Claim] = {}
        self.objects: Dict[str, ObjectHypothesis] = {}
        self.candidate_edges: List = []
        self.relations: Dict[str, Relation] = {}
        self.provenance = ProvenanceGraph(self.config)
        self.inference_state: Optional[InferenceState] = None
        self.coverage: Dict[str, int] = {}
        self.ingest_log: List[Dict[str, Any]] = []
        self.duplicate_ingests: int = 0
        self._call_fingerprints: set = set()
        # call fingerprint -> the first observation id produced by that call, used
        # as the duplicate owner when a replay is detected.
        self._call_owners: Dict[str, str] = {}
        # Evidence revision counters: any ingest or object registration bumps
        # ``_revision``; a cached inference is only valid while the revision it
        # was computed from matches. Without this, a second acquisition could
        # read a stale posterior that predates the new evidence.
        self._revision: int = 0
        self._inference_revision: int = -1
        # Whether ``build_problem`` may mint hypotheses from observation
        # clusters. Frozen evaluators register the annotated object set
        # explicitly and must stay bit-identical, so they disable autoseeding;
        # an accreting acquisition store leaves it on to discover new objects.
        self.autoseed_objects: bool = True

    # ---------------------------------------------------------------- ingest

    def ingest_tool_response(
        self,
        tool_name: str,
        response: Mapping[str, Any],
        input_registry: Optional[Mapping[str, Any]] = None,
        round_id: Optional[int] = None,
        api_args: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """Ingest one tool response, returning the new observation ids.

        Failed responses contribute nothing.  Exact replays of an already-seen
        (tool, normalized-input) pair register provenance duplicates so their
        claims receive zero weight instead of amplifying the posterior.
        """
        adapter = adapter_pkg.resolve(tool_name)
        if adapter is None:
            self.ingest_log.append(
                {"tool": tool_name, "round_id": round_id, "skipped": "no adapter"}
            )
            return []

        context = self._build_context(input_registry, round_id, api_args)
        result = adapter.to_observations(tool_name, response, context)
        if result.skipped_reason and not result.observations:
            self.ingest_log.append(
                {"tool": tool_name, "round_id": round_id, "skipped": result.skipped_reason}
            )
            return []

        # Exact-replay detection operates on the whole tool call, independent of
        # round_id, so re-running an identical request is recognized as a replay.
        call_fingerprint = compute_evidence_fingerprint(
            tool_name,
            dict(api_args or {}),
            parameters={"adapter": type(adapter).__name__},
            input_roots=list(context.get("input_roots") or []),
        )
        # A replay is the same tool invoked with the same normalized inputs. The
        # response payload is deliberately not part of the call fingerprint, so a
        # nondeterministic tool that jitters its boxes is still recognized as a
        # replay. Anything it produces beyond the first call must carry zero
        # weight, otherwise re-issuing one request inflates the posterior.
        replay = call_fingerprint in self._call_fingerprints
        replay_owner = self._call_owners.get(call_fingerprint) if replay else None
        if replay:
            self.duplicate_ingests += 1
        else:
            self._call_fingerprints.add(call_fingerprint)

        new_ids: List[str] = []
        for observation in result.observations:
            existing = self.observations.get(observation.observation_id)
            if existing is None:
                self.observations[observation.observation_id] = observation
                new_ids.append(observation.observation_id)
            else:
                for claim_id in observation.claim_ids:
                    if claim_id not in existing.claim_ids:
                        existing.claim_ids.append(claim_id)
            if observation.observation_id not in self.provenance.nodes:
                self.provenance.register(
                    node_id=observation.observation_id,
                    kind="observation",
                    tool_name=str(tool_name),
                    fingerprint_value=f"{call_fingerprint}:{observation.observation_id}",
                    parents=list(observation.derivation_parents),
                    input_roots=list(observation.input_roots),
                    round_id=round_id,
                    duplicate_of=replay_owner,
                )
            self._call_owners.setdefault(call_fingerprint, observation.observation_id)

        new_claims = 0
        for claim in result.claims:
            claim_fingerprint = compute_evidence_fingerprint(
                tool_name,
                {
                    "observation": claim.observation_id,
                    "attribute": claim.attribute,
                    "value": claim.value,
                    "time": claim.time_scope.key,
                },
                input_roots=list(
                    self.observations[claim.observation_id].input_roots
                    if claim.observation_id in self.observations
                    else []
                ),
            )
            if claim.claim_id in self.claims:
                # Identical claim seen again: keep the original, do not re-weight.
                continue
            self.claims[claim.claim_id] = claim
            new_claims += 1
            self.provenance.register(
                node_id=claim.claim_id,
                kind="claim",
                tool_name=str(tool_name),
                fingerprint_value=claim_fingerprint,
                parents=[claim.observation_id],
                input_roots=list(
                    self.observations[claim.observation_id].input_roots
                    if claim.observation_id in self.observations
                    else []
                ),
                round_id=round_id,
                # Claims produced by a replayed call get zero weight, so a
                # nondeterministic tool cannot amplify belief by being re-run.
                duplicate_of=replay_owner,
            )

        new_relations = 0
        for relation in result.relations:
            if relation.relation_id not in self.relations:
                self.relations[relation.relation_id] = relation
                new_relations += 1

        # Coverage is recomputed from the deduplicated observation set so that a
        # replay cannot inflate it.
        self._recompute_coverage()

        if new_ids or new_claims or new_relations:
            self._revision += 1

        self.ingest_log.append(
            {
                "tool": tool_name,
                "round_id": round_id,
                "observations": len(result.observations),
                "claims": len(result.claims),
                "relations": len(result.relations),
                "new_observations": len(new_ids),
                "replay": replay,
                "revision": self._revision,
            }
        )
        return new_ids

    def _recompute_coverage(self) -> Dict[str, int]:
        """Rebuild ``coverage`` as counts of distinct observations per cell.

        Replayed observations are excluded. Coverage answers "how much
        independent looking have we done", so counting a re-run of the same call
        would make thin evidence look well covered and let the decision layer
        clear its ``min_coverage`` gate without any new information.
        """
        coverage: Dict[str, int] = {}
        for observation in self.observations.values():
            if self.provenance.is_duplicate(observation.observation_id):
                continue
            key = f"{observation.modality.value}|{observation.time_scope.key}"
            coverage[key] = coverage.get(key, 0) + 1
        self.coverage = coverage
        return coverage

    def _build_context(
        self,
        input_registry: Optional[Mapping[str, Any]],
        round_id: Optional[int],
        api_args: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        registry = dict(input_registry or {})
        args = dict(api_args or {})
        primary = args.get("image") or registry.get("primary_image")
        optical = args.get("rgb_image") or args.get("optical_image") or registry.get("optical_image")
        sar = args.get("sar_image") or registry.get("sar_image")
        before_image = args.get("before_image") or registry.get("before_image") or primary
        after_image = args.get("after_image") or registry.get("after_image")
        roots = [value for value in {primary, optical, sar} if value]
        if optical and sar:
            relation_id = f"cross_modal:{optical}:{sar}"
            image_id = f"cross_modal:optical:{optical}"
            related_ids = [f"cross_modal:sar:{sar}"]
        elif before_image and after_image:
            before = before_image
            after = after_image
            relation_id = f"time:{before}:{after}"
            image_id = f"time:before:{before}"
            related_ids = [f"time:after:{after}"]
        elif primary:
            relation_id = f"single:{primary}"
            image_id = f"single:{primary}"
            related_ids = []
        else:
            relation_id = None
            image_id = None
            related_ids = []
        return {
            "round_id": round_id,
            "primary_image": primary,
            "optical_image": optical,
            "sar_image": sar,
            "acquisition_id": optical or primary or sar,
            "pixel_size_m": args.get("pixel_size_m") or registry.get("pixel_size_m"),
            "time_key": str(args.get("time_key") or "t0"),
            "before_image": before_image,
            "after_image": after_image,
            "input_roots": roots,
            "relation_group_id": relation_id,
            "image_group_id": image_id,
            "related_image_group_ids": related_ids,
        }

    # ---------------------------------------------------------- object layer

    def seed_object_hypotheses(self, iou_merge_threshold: float = 0.5) -> List[str]:
        """Propose object hypotheses by clustering current observation supports.

        Purely a bootstrap for candidate-graph columns; the clustering makes no
        identity commitment.  Explicitly registered objects are preserved, and a
        seeded cluster that overlaps an existing hypothesis is treated as that
        same object: its id depends on cluster order and mean box, so adding a
        view would otherwise mint a duplicate of an object already present.
        """
        from DS_VI.models.shared_identity.physical_features import bbox_iou

        association_cfg = dict(self.config.get("association") or {})
        if str(association_cfg.get("mode", "legacy_global")) == (
            "cross_image_independent"
        ):
            from DS_VI.models.shared_identity.observation_builder import (
                cross_image_object_hypotheses,
            )

            seeded_hypotheses = cross_image_object_hypotheses(
                list(self.observations.values()),
                None
                if association_cfg.get("pair_capacity") is None
                else int(association_cfg["pair_capacity"]),
            )
            seeded = [
                (hypothesis.object_id, hypothesis.spatial_support.bbox, hypothesis)
                for hypothesis in seeded_hypotheses
            ]
        else:
            seeded = [
                (object_id, bbox, None)
                for object_id, bbox in object_hypotheses_from_observations(
                    list(self.observations.values()), iou_merge_threshold
                )
            ]
        raw_default_capacity = dict(self.config.get("inference") or {}).get(
            "default_capacity"
        )
        default_capacity = (
            None if raw_default_capacity is None else int(raw_default_capacity)
        )
        created: List[str] = []
        for object_id, bbox, seeded_hypothesis in seeded:
            if object_id in self.objects:
                continue
            matched = any(
                existing.spatial_support.bbox
                and bbox_iou(existing.spatial_support.bbox, bbox) >= iou_merge_threshold
                for existing in self.objects.values()
            )
            if matched:
                continue
            hypothesis = seeded_hypothesis or ObjectHypothesis(
                object_id=object_id,
                spatial_support=SpatialSupport(bbox=bbox),
                capacity=default_capacity,
                context={"origin": "seeded_from_observations"},
            )
            if hypothesis.capacity is None:
                hypothesis.capacity = default_capacity
            hypothesis.validate()
            self.objects[object_id] = hypothesis
            created.append(object_id)
        if created:
            self._revision += 1
        return created

    def register_object(self, hypothesis: ObjectHypothesis) -> str:
        """Register an externally supplied object hypothesis."""
        hypothesis.validate()
        self.objects[hypothesis.object_id] = hypothesis
        self._revision += 1
        return hypothesis.object_id

    def rebuild_candidate_graph(self) -> List:
        """Recompute the candidate graph from current observations and objects."""
        self.candidate_edges = build_candidate_graph(
            list(self.observations.values()),
            list(self.objects.values()),
            self.config,
        )
        return self.candidate_edges

    # ------------------------------------------------------------- inference

    def build_problem(self, seed_objects: bool = True) -> InferenceProblem:
        """Assemble an :class:`InferenceProblem` from the current state."""
        if seed_objects and self.autoseed_objects:
            # Seed on every build, not only when empty: evidence acquired in a
            # later round can introduce an object the first round never saw.
            self.seed_object_hypotheses()
        if not self.candidate_edges or seed_objects:
            self.rebuild_candidate_graph()

        # Paper Eq.(29) constrains capacity per acquisition/channel block only;
        # a missing value means "no column cap" rather than a default of 2.
        default_capacity = dict(self.config.get("inference") or {}).get("default_capacity")
        capacities = {}
        for object_id, obj in self.objects.items():
            value = obj.capacity if obj.capacity is not None else default_capacity
            if value is not None:
                capacities[object_id] = int(value)
        return InferenceProblem(
            observations=list(self.observations.values()),
            claims=list(self.claims.values()),
            objects=list(self.objects.values()),
            candidate_edges=list(self.candidate_edges),
            relations=list(self.relations.values()),
            capacities=capacities,
        )

    def run_inference(self, seed_objects: bool = True) -> InferenceState:
        """Run joint A-Q inference over the accumulated evidence."""
        from DS_VI.models.shared_identity.inference import run_joint_inference

        problem = self.build_problem(seed_objects=seed_objects)
        self.inference_state = run_joint_inference(
            problem, self.config, provenance_graph=self.provenance
        )
        self._inference_revision = self._revision
        return self.inference_state

    def build_evidence_state(self):
        """Summarize the store, re-running inference when evidence changed.

        A cached posterior is only valid for the evidence revision it was
        computed from; returning it after a later acquisition would report the
        old state as if the new evidence had been considered.
        """
        from DS_VI.models.shared_identity.evidence_state import build_evidence_state

        if self.inference_state is None or self._inference_revision != self._revision:
            self.run_inference()
        return build_evidence_state(self, self.inference_state)

    # --------------------------------------------------------------- summary

    def stats(self) -> Dict[str, Any]:
        return {
            "observations": len(self.observations),
            "claims": len(self.claims),
            "objects": len(self.objects),
            "candidate_edges": len(self.candidate_edges),
            "relations": len(self.relations),
            "coverage": dict(self.coverage),
            "duplicate_ingests": self.duplicate_ingests,
            "provenance": self.provenance.summarize(),
            "inference_ran": self.inference_state is not None,
        }

    def claims_for_observation(self, observation_id: str) -> List[Claim]:
        return [c for c in self.claims.values() if c.observation_id == observation_id]

    def to_dict(self) -> Dict[str, Any]:
        from DS_VI.models.shared_identity.serialization import store_to_dict

        return store_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceStore:
        from DS_VI.models.shared_identity.serialization import store_from_dict

        return store_from_dict(payload, cls)


__all__ = ["EvidenceStore"]
