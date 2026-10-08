"""M6.8 fixed-ranking breadth and duplicate-x validation tests."""
from __future__ import annotations
import pytest
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.characterization import rank_controlled_predictor
from edge_moe_sim.plot_validation import assert_unique_x_per_line

def test_progressive_fixed_ranking_recall_and_nested_candidates() -> None:
    event=BatchRoutingEvent(1,0,0,((0,1,2,3),))
    predictor,ranking,trace_id=rank_controlled_predictor(event,32,true_ranks_one_based=(1,3,7,13))
    true={0,1,2,3}; recalls=[];sets=[]
    for width,expected in ((2,.25),(4,.5),(8,.75),(16,1.0)):
        candidates={x.expert_id for x in predictor.predict(event).aggregate_candidates(width)};sets.append(candidates);recalls.append(len(candidates&true)/4);assert recalls[-1]==expected
    assert all(left<=right for left,right in zip(sets,sets[1:]));assert trace_id.startswith('rank-controlled') and ranking[0]==0

def test_unique_x_rejects_duplicates(tmp_path) -> None:
    with pytest.raises(ValueError,match='duplicate'):
        assert_unique_x_per_line([{'x':1,'line':'a','other':1},{'x':1,'line':'a','other':2}],x_column='x',line_group_columns=['line'],figure_name='test',debug_dir=tmp_path)
