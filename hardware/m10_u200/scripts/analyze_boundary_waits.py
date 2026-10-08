"""被动事件分段：区分可比较的抽象等待与 RTL 独有接口步骤。"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


def analyze_waits(log: str, result: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    """写出事件及逐 tile 非重叠 pre-GU 等待，时间单位均为 cycle。"""
    controls = [dict(event=e, layer=int(l), cycle=int(c), expert=int(x), tile=int(t),
                     matrix=int(m), slot=int(s), generation=int(g), epoch=int(p))
        for e,l,c,x,t,m,s,g,p in re.findall(
            r'MAIN_CTL event=(\w+) layer=(\d+) cycle=(\d+) expert=(\d+) tile=(\d+) matrix=(\d+) slot=(\d+) generation=(\d+) epoch=(\d+)', log)]
    if not controls:
        raise ValueError('缺少带 ownership 标签的 MAIN_CTL 被动观测')
    with (directory/'control_events.csv').open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(controls[0]));writer.writeheader();writer.writerows(controls)
    credits=[tuple(map(int, values)) for values in re.findall(
        r'MAIN_CREDIT cycle=(\d+) descriptor_free=(\d+) occupancy=(\d+) demand_pair=(\d+) spec_pair=(\d+)',log)]
    with (directory/'credit_events.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['cycle','descriptor_free','occupancy','demand_pair','spec_pair']);writer.writerows(credits)
    blockers=re.findall(r'MAIN_BLOCK cycle=(\d+) layer=(\d+) pending=(\d+) stale=(\d+) combine1=(\d+) combine2=(\d+) gu=(\d+) down=(\d+)',log)
    with (directory/'scheduler_blockers.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['cycle','layer','pending','stale','combine1','combine2','gu_found','down_found']);writer.writerows(blockers)
    stages=[(int(l),int(c),int(x),int(t),s) for l,c,x,t,s in re.findall(
        r'MAIN_STAGE layer=(\d+) cycle=(\d+) expert=(\d+) tile=(\d+) stage=(\w+)',log)]
    starts={int(l):int(c) for l,c in re.findall(r'LAYER_START layer=(\d+) cycle=(\d+)',log)}
    truths={l:max(int(c) for layer,c in re.findall(r'MAIN_TRUTH layer=(\d+) cycle=(\d+)',log) if int(layer)==l) for l in starts}
    trace=result['trace']; rows: list[dict[str,Any]]=[]

    def control(event: str, key: tuple[int,int,int], matrix: int=0) -> int:
        """要求每个物理 tile/matrix 的关键事件唯一，不能静默丢失。"""
        matches=[v for v in controls if v['event']==event and (v['layer'],v['expert'],v['tile'])==key and v['matrix']==matrix]
        if len(matches)!=1:raise ValueError(f'关键事件缺失/重复：{event} {key} {matrix}')
        return matches[0]['cycle']

    def main_cycle(event: str, key: tuple[int,int,int], matrix: str='gate_up') -> int:
        """找到主模型事件；pair 以逻辑原子 admission 表达。"""
        matches=[v for v in trace if v['event_type']==event and (v.get('layer_id'),v.get('expert_id'),v.get('tile_id'))==key
                 and (event=='pair_admit' or v.get('matrix_type')==matrix)]
        if len(matches)!=1:raise ValueError(f'主模型事件缺失/重复：{event} {key}')
        return matches[0]['cycle']

    for layer,begin in sorted(starts.items()):
        model_layer=result['layer_results'][layer]; origin_r=begin;origin_m=model_layer['layer_start_cycle']
        for _,gu_start,expert,tile,_ in (v for v in stages if v[0]==layer and v[4]=='GU_START'):
            key=(layer,expert,tile)
            ready_r=control('READY',key); down_r=control('ADMIT',key,1)
            selected_r=control('SELECT',key); accept_r=control('COMPUTE_ACCEPT',key)
            ready_event = 'payload_ready' if any(v['event_type']=='payload_ready'
                and (v.get('layer_id'),v.get('expert_id'),v.get('tile_id'))==key
                and v.get('matrix_type')=='gate_up' for v in trace) else 'dma_done'
            ready_m=main_cycle(ready_event,key); down_m=main_cycle('pair_admit',key)
            selected_m=main_cycle('tile_select',key); start_m=main_cycle('aie_start',key)
            dep_r=max(origin_r,truths[layer],down_r);dep_m=max(origin_m,model_layer['router_resolve_cycle'],down_m)
            eligible_r=max(dep_r,ready_r);eligible_m=max(dep_m,ready_m)
            if not eligible_r<=selected_r<=accept_r<=gu_start:raise ValueError(f'RTL 因果顺序错误 {key}')
            if not eligible_m<=selected_m<=start_m:raise ValueError(f'主模型因果顺序错误 {key}')
            ready_class='before_truth' if ready_r<truths[layer] else 'at_truth' if ready_r==truths[layer] else 'after_truth'
            intervals=[('dependency_wait',dep_m-origin_m,dep_r-origin_r),
                ('memory_wait',eligible_m-dep_m,eligible_r-dep_r),
                ('eligible_to_gu',start_m-eligible_m,gu_start-eligible_r),
                ('eligible_to_select',selected_m-eligible_m,selected_r-eligible_r),
                ('select_to_accept',None,accept_r-selected_r),('accept_to_gu',None,gu_start-accept_r)]
            for name,m,r in intervals:
                rows.append(dict(layer=layer,expert=expert,tile=tile,interval=name,main_cycles=m,rtl_cycles=r,
                    delta_cycles=None if m is None else r-m,ready_class=ready_class,
                    partition_member=name in ('dependency_wait','memory_wait','eligible_to_gu'),
                    semantics='abstract_dispatch_unsplit' if m is None else 'comparable_aggregate'))
            for matrix,event in ((0,'RELEASE_GU'),(1,'RELEASE_DOWN')):
                released=control(event,key,matrix);free=control('SLOT_FREE',key,matrix)
                if free<released:raise ValueError(f'credit 早于 release {key} {matrix}')
                rows.append(dict(layer=layer,expert=expert,tile=tile,interval=f'{event.lower()}_to_free',
                    main_cycles=0,rtl_cycles=free-released,delta_cycles=free-released,ready_class=ready_class,
                    partition_member=False,semantics='logical_vs_registered_slot_credit'))
            # origin 为前一个 tile DOWN 完成，三个 partition 成员严格相加到当前 GU。
            assert dep_r-origin_r+eligible_r-dep_r+gu_start-eligible_r==gu_start-origin_r
            origin_r=next(c for l,c,x,t,s in stages if (l,x,t,s)==(*key,'DOWN_DONE'))
            done=[v for v in trace if v['event_type']=='aie_start' and (v.get('layer_id'),v.get('expert_id'),v.get('tile_id'),v.get('matrix_type'))==(*key,'down')]
            origin_m=done[0]['end_cycle']
    with (directory/'wait_segments.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows
