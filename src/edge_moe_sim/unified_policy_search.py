"""Bounded policy-plan generation and deterministic winner selection for M7.2."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable,Mapping,Sequence

@dataclass(frozen=True)
class UnifiedPolicyPlan:
    """One bounded GU-prefix decision over a fixed candidate ranking."""
    family:str; candidate_ids:tuple[int,...]; depths:tuple[int,...]
    @property
    def candidate_count(self)->int:return len(self.candidate_ids)
    @property
    def signature(self)->str:return f"{self.candidate_ids}:{self.depths}"
    @property
    def mean_depth(self)->float:return sum(self.depths)/len(self.depths) if self.depths else 0.0
    @property
    def depth_concentration(self)->float:return max(self.depths,default=0)/sum(self.depths) if sum(self.depths) else 0.0

class UnifiedBreadthDepthSearch:
    """Generate a small deduplicated set of uniform/rank-weighted/mixed plans."""
    def __init__(self,tile_bytes:int,max_tiles:int=8,breadths:Sequence[int]=(1,2,4,8,16),depths:Sequence[int]=(0,1,2,4,8)) -> None:
        self.tile_bytes,self.max_tiles,self.breadths,self.depths=tile_bytes,max_tiles,tuple(breadths),tuple(depths)
    def plans_with_stats(self,ranking:Sequence[int],budget_per_prediction:int|None=None)->tuple[tuple[UnifiedPolicyPlan,...],dict[str,int]]:
        """Return deduplicated feasible plans and the pre-dedup attempt count."""
        unique={}
        generated=0;static_pruned=0;duplicate_pruned=0
        for m in self.breadths:
            if m>len(ranking):continue
            ids=tuple(ranking[:m])
            vectors=[]
            for d in self.depths:vectors.append(("uniform",tuple(min(d,self.max_tiles) for _ in ids)))
            for d in self.depths:vectors.append(("rank_weighted",tuple(min(self.max_tiles,max(1,d//(1+i//2))) for i in range(m))))
            if m>1:vectors.append(("mixed",tuple(max(1,min(self.max_tiles,8//(i+1))) for i in range(m))))
            for family,vector in vectors:
                generated+=1
                # The all-zero vector is exactly the no-prefetch request plan and
                # is already represented by P0, so prune it before simulation.
                if not any(vector):static_pruned+=1;continue
                allocated=sum(vector)*self.tile_bytes
                if budget_per_prediction is not None and allocated>budget_per_prediction:static_pruned+=1;continue
                plan=UnifiedPolicyPlan(family,ids,vector)
                if plan.signature in unique:duplicate_pruned+=1;continue
                unique[plan.signature]=plan
        plans=tuple(sorted(unique.values(),key=lambda p:(p.candidate_count,p.depths,p.family)))
        return plans,{'generated':generated,'unique':len(plans),'static_pruned':static_pruned,'duplicate_pruned':duplicate_pruned}

    def plans(self,ranking:Sequence[int],budget_per_prediction:int|None=None)->tuple[UnifiedPolicyPlan,...]:
        """Return feasible request-identity plans with exact duplicates removed."""
        return self.plans_with_stats(ranking,budget_per_prediction)[0]

def rank_policy_rows(rows:Iterable[Mapping[str,object]],tolerance:float=0.0)->list[dict[str,object]]:
    """Rank one configuration's rows by makespan then canonical cost tie-breakers."""
    data=[dict(row) for row in rows]
    if not data:return []
    best=min(float(r["sequence_makespan"]) for r in data)
    for row in data:
        latency=float(row["sequence_makespan"]);row["regret_cycles"]=latency-best;row["regret_percent"]=latency/best-1
    data.sort(key=lambda r:(float(r["sequence_makespan"]),float(r.get("wasted_physical_speculative_bytes",0)),float(r.get("physical_speculative_bytes_transferred_by_router",0)),float(r.get("initial_allocated_speculative_bytes",0)),float(r.get("peak_ocm_bytes",0)),str(r.get("policy_plan_signature",""))))
    for i,row in enumerate(data,1):row["search_rank"]=i
    return data
