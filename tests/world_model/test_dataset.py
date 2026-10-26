import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from test_preprocess import fixture
from pubg_lab.world_model.data.preprocess import build_bank
from pubg_lab.world_model.data.dataset import WorldDataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        for d in ("banks","meta","ablations"):
            (self.root/d).mkdir()
        rows=[]
        for mid,role,patch,item in [("fit","train","40.2","FitGun"),
                                    ("excluded","train","40.2","ExcludedGun"),
                                    ("dev","validation","41.1","UnseenGun")]:
            raw=fixture()
            for r in raw["tables"]["equip"]:
                r["item"]=item
            b,m=build_bank(raw)
            np.savez_compressed(self.root/"banks"/f"{mid}.npz",**b)
            (self.root/"meta"/f"{mid}.json").write_text(json.dumps(m))
            rows.append({"match_id":mid,"role":role,"patch":patch,"map":"A",
                         "perspective":"TPP","nominal_team_size":"5","game_mode":"squad",
                         "geometry_key":"A__"+patch})
        with (self.root/"matches.csv").open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        plan={"train_match_ids":["fit"],"validation_match_ids":["dev"]}
        for name in ("full","server","no_team_node","no_spatial","no_outcome_loss"):
            (self.root/"ablations"/f"{name}.json").write_text(json.dumps(plan))
        db=sqlite3.connect(self.root/"catalog.sqlite")
        db.execute("CREATE TABLE spatial_support(match_id,geometry_key,sx,sy,dx,dy,mode,n)")
        db.executemany("INSERT INTO spatial_support VALUES(?,?,?,?,?,?,?,?)",
                       [("fit","A__40.2",1,1,2,1,1,3),
                        ("excluded","A__40.2",1,1,2,1,1,99)])
        db.commit();db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_fold_specific_vocabulary_support_and_unknown_patch(self):
        ds=WorldDataset(self.root,role="validation")
        self.assertEqual(set(ds.item_vocabulary),{"FitGun"})
        counts,matches=ds.support("A__40.2")
        self.assertEqual(counts[1,1,2,1,1],3)
        self.assertEqual(matches[1,1,2,1,1],1)
        g,y=ds.sample("dev",0)
        self.assertTrue((g["items_sparse"][:,2]==1).all())
        self.assertEqual(len(g["observed_connections"]),0)
        self.assertTrue((g["cell_features"][:,5]==0).all())
        with self.assertRaises(PermissionError):
            ds.bank("fit")

    def test_server_selector_and_same_targets(self):
        obs=WorldDataset(self.root,role="validation")
        server=WorldDataset(self.root,ablation="server",role="validation")
        a,y=obs.sample("dev",0)
        b,z=server.sample("dev",0)
        self.assertEqual(len(a["player_ids"]),5)
        self.assertEqual(len(b["player_ids"]),6)
        np.testing.assert_array_equal(y["xy"],z["xy"])
        np.testing.assert_array_equal(a["cell_coordinates"],b["cell_coordinates"])
        self.assertTrue((a["reentry_confidence"][:,1:]==0).all())

    def test_ablations_really_remove_the_requested_component(self):
        g,y=WorldDataset(self.root,ablation="no_team_node",role="validation").sample("dev",0)
        self.assertEqual(len(g["team_ids"]),0)
        self.assertEqual(len(g["member_edges"]),0)
        g,y=WorldDataset(self.root,ablation="no_spatial",role="validation").sample("dev",0)
        self.assertEqual(len(g["cell_features"]),0)
        self.assertEqual(len(g["located_in"]),0)
        g,y=WorldDataset(self.root,ablation="no_outcome_loss",role="validation").sample("dev",0)
        self.assertTrue(all(not t["rank_mask"] and not t["winner_mask"] for t in y["outcomes"]))


if __name__=="__main__":
    unittest.main()
