import copy
import unittest
import numpy as np
from pubg_lab.world_model.data.preprocess import build_bank, view_at, targets_at, outcomes
from pubg_lab.world_model.data.patches import infer_patch, at_time


def fixture():
    raw = {"meta":{"br_rule_candidate":True,"end_time":"known"},
           "actors":[{"actor":p,"team":0 if p<5 else 1,"kind":1,"rank":1 if p<5 else 2}
                     for p in range(6)],
           "teams":[{"team":0,"rank":1,"won":True},{"team":1,"rank":2,"won":False}],
           "tables":{k:[] for k in ["positions","landing","kills","damage","equip","pickup","use","game"]}}
    for p in range(6):
        raw["tables"]["landing"].append({"actor":p,"t":0})
        for t in range(0,241,10):
            raw["tables"]["positions"].append({"actor":p,"t":t,"row_id":p*100+t,
                "x":100+p*3+t,"y":100+p,"z":10,"hp":100,"vehicle_type":None,"vehicle_class":None})
        raw["tables"]["equip"].append({"actor":p,"t":5,"item":"Weapon_B",
                                      "action":"equip","category":"Weapon"})
    for t in range(0,241,10):
        raw["tables"]["game"].append({"t":t,"phase":1,"alive_players":6,"alive_teams":2,
                     "safe_x":100,"safe_y":100,"safe_r":500,"poison_x":100,"poison_y":100,"poison_r":900})
    return raw


class PreprocessTests(unittest.TestCase):
    def test_post_match_end_logs_do_not_become_training_states(self):
        r=fixture()
        r["meta"]["end_s"]=125.
        bank,meta=build_bank(r)
        self.assertLessEqual(float(bank["times"][-1]),125.)
        self.assertGreater(meta["post_end_rows_excluded"]["positions"],0)
        self.assertFalse(targets_at(bank,12,0)["mask"].any())
        self.assertTrue(any(v[3]=="post_match_end_positions" for v in meta["issues"]))

    def test_patch_schedule_boundary_timezone_and_maintenance(self):
        self.assertEqual(at_time("2026-04-08T08:59:59+09:00"), "40.2")
        self.assertIsNone(at_time("2026-04-08T09:00:00+09:00"))
        self.assertEqual(at_time("2026-04-08T17:30:00+09:00"), "41.1")
        self.assertEqual(at_time("2026-06-09T00:00:00+00:00"), "41.2")
        row={"shard_id":"steam","created_at":"2026-04-07T23:50:00+00:00",
             "start_time":"2026-04-07T23:55:00+00:00","end_time":"2026-04-08T00:10:00+00:00"}
        self.assertEqual(infer_patch(row)[0],"unknown")

    def test_causal_prefix_and_future_outcome_invariance(self):
        r=fixture()
        full,meta=build_bank(r)
        short=copy.deepcopy(r)
        short["tables"]={k:[v for v in rs if v["t"]<=120] for k,rs in r["tables"].items()}
        partial,_=build_bank(short)
        for key in ["x","mask","valid","source_time","known","episode","segment",
                    "mobility","confidence","game","game_mask","game_time"]:
            np.testing.assert_array_equal(full[key][:13],partial[key],err_msg=key)
        r["teams"][0]["rank"]=2
        r["tables"]["equip"].append({"actor":0,"t":200,"item":"Weapon_A","action":"equip"})
        changed,_=build_bank(r)
        a,b=view_at(full,12,0),view_at(changed,12,0)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)

    def test_observer_private_invariance_before_team_aggregation(self):
        bank,_=build_bank(fixture())
        altered={k:v.copy() for k,v in bank.items()}
        altered["x"][:,1,8:]=99999  # teammate private state
        altered["x"][:,5,:]=99999  # enemy server-only state
        a,b=view_at(bank,12,0),view_at(altered,12,0)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        self.assertFalse(np.array_equal(view_at(bank,12,0,"server")["x"],
                                        view_at(altered,12,0,"server")["x"]))

    def test_variable_team_no_truncation_and_joint_distance(self):
        bank,_=build_bank(fixture())
        view=view_at(bank,12,0)
        y=targets_at(bank,12,0)
        self.assertEqual(len(view["player_ids"]),5)
        self.assertEqual(y["xy"].shape,(5,6,2))
        self.assertTrue(y["mask"].all())
        self.assertAlmostEqual(float(np.linalg.norm(y["xy"][1,0]-y["xy"][0,0])),10**.5,places=5)

    def test_reentry_and_death_do_not_bridge(self):
        r=fixture()
        r["tables"]["kills"]=[{"actor":0,"t":130,"rank":None}]
        for x in r["tables"]["positions"]:
            if x["actor"]==0 and x["t"]>=150:
                x["vehicle_type"]="RedeployAircraft"
        b,m=build_bank(r)
        self.assertFalse(targets_at(b,12,0)["mask"][0].any())
        self.assertEqual(int(b["episode"][15,0]),1)
        self.assertEqual(int(b["mobility"][15,0]),5)
        self.assertFalse(any(p==0 for k,p,item in b["items_sparse"] if k>=13))

    def test_gap_and_conflicting_timestamp(self):
        r=fixture()
        r["tables"]["positions"]=[x for x in r["tables"]["positions"]
                                  if not(x["actor"]==0 and 130<=x["t"]<=150)]
        dup=copy.deepcopy(next(x for x in r["tables"]["positions"] if x["actor"]==1 and x["t"]==120))
        dup["x"]+=100
        r["tables"]["positions"].append(dup)
        b,m=build_bank(r)
        self.assertFalse(b["valid"][12,1])
        self.assertFalse(targets_at(b,12,0)["mask"][0].any())
        self.assertTrue(any(i[3]=="track_gap" for i in m["issues"]))

    def test_outcomes_never_resolve_conflict_by_overwrite(self):
        r=fixture()
        self.assertTrue(all(x["rank_mask"] for x in outcomes(r)))
        r["actors"][0]["rank"]=2
        v=outcomes(r)[0]
        self.assertFalse(v["rank_mask"])
        self.assertFalse(v["winner_mask"])
        self.assertEqual(v["rank"],1)
        self.assertFalse(v["terminal_time_mask"])

    def test_future_actor_absent_from_graph(self):
        r=fixture()
        r["tables"]["positions"]=[x for x in r["tables"]["positions"]
                                  if x["actor"]!=4 or x["t"]>=180]
        b,_=build_bank(r)
        self.assertNotIn(4,view_at(b,12,0)["player_ids"])
        self.assertNotIn(4,targets_at(b,12,0)["members"])

    def test_late_actor_placeholder_is_not_a_prefix_input(self):
        r=fixture()
        r["tables"]["positions"]=[x for x in r["tables"]["positions"]
                                  if x["actor"]!=4 or x["t"]>=180]
        full,_=build_bank(r)
        short=copy.deepcopy(r)
        short["tables"]={k:[v for v in rs if v["t"]<=120]
                         for k,rs in r["tables"].items()}
        prefix,_=build_bank(short)
        np.testing.assert_array_equal(
            np.where(full["known"][:13],full["confidence"][:13],0),
            np.where(prefix["known"],prefix["confidence"],0))
        for visibility in ("observer","server"):
            a,b=view_at(full,12,0,visibility),view_at(prefix,12,0,visibility)
            for key in a:
                np.testing.assert_array_equal(a[key],b[key],err_msg=key)


if __name__=="__main__":
    unittest.main()
