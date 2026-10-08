"""M6.6 controlled predictor and plotting-validation tests."""
from __future__ import annotations
import pytest
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.characterization import controlled_predictor
from edge_moe_sim.plot_validation import require_plot_data, validate_line_group

def test_perfect_and_wrong_controlled_predictors_have_expected_ranking() -> None:
    event=BatchRoutingEvent(7,0,0,((0,1),(0,1)))
    perfect,_,rec,any_hit,full=controlled_predictor(event,8,profile='perfect',candidate_count=4,ready_cycle=0)
    wrong,_,wrong_rec,wrong_any,wrong_full=controlled_predictor(event,8,profile='wrong',candidate_count=4,ready_cycle=0)
    assert rec==1 and any_hit and full
    assert wrong_rec==0 and not wrong_any and not wrong_full
    assert {x.expert_id for x in perfect.predict(event).aggregate_candidates(2)}=={0,1}
    assert not ({x.expert_id for x in wrong.predict(event).aggregate_candidates(4)} & {0,1})

def test_plot_validation_rejects_blank_and_uncontrolled_lines() -> None:
    with pytest.raises(ValueError,match='empty'): require_plot_data([],['x'])
    with pytest.raises(ValueError,match='uncontrolled'):
        validate_line_group([{'x':1,'bandwidth':1},{'x':2,'bandwidth':2}],sweep='x',fixed=['bandwidth'])
