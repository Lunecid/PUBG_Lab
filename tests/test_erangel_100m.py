"""Leakage and trajectory-boundary regression tests for the spatial approximation."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.spatial.erangel.geometry import erangel_100m_build as build
from projects.spatial.erangel.ingestion.erangel_100m_extract import export_match


class Cursor:
    def __init__(self, replies):self.replies=iter(replies)
    def execute(self,*args):self.current=next(self.replies)
    def fetchall(self):return self.current


class TrajectoryBoundaryTests(unittest.TestCase):
    def setUp(self):self.t=datetime(2026,3,27,tzinfo=timezone.utc)
    def row(self,second,x,account='account.a',hp=100,vehicle=None):
        return account,self.t+timedelta(seconds=second),x*100,10000,100,hp,vehicle,None
    def extract(self,positions,landing=None,deaths=()):
        cur=Cursor([[('account.a',),('account.b',),('ai.bot',)],landing or [('account.a',self.t),('account.b',self.t),('ai.bot',self.t)],deaths,positions])
        return export_match(cur,{'match_id':'synthetic'},np.full((8,8),32768,dtype=np.float32))
    def test_accounts_never_connect_and_kinds_survive(self):
        nodes,edges,counts=self.extract([self.row(0,50),self.row(10,150),self.row(20,250,'account.b'),self.row(30,350,'account.b'),self.row(40,450,'ai.bot'),self.row(50,550,'ai.bot')])
        self.assertEqual(int(edges[:,4].sum()),3)
        self.assertEqual(set(edges[:,2]),{1,2})
        self.assertNotIn((83,84),{tuple(e[:2]) for e in edges})
    def test_rejected_row_breaks_continuity(self):
        _,edges,c=self.extract([self.row(0,50),self.row(5,100,hp=0),self.row(10,150),self.row(20,250)])
        self.assertEqual(int(edges[:,4].sum()),1)
        self.assertEqual(c['excluded_not_alive'],1)
    def test_duplicates_are_all_excluded_without_bridging(self):
        _,edges,c=self.extract([self.row(0,50),self.row(5,100),self.row(5,110),self.row(10,150)])
        self.assertEqual(len(edges),0);self.assertEqual(c['excluded_duplicate_timestamp'],2)
    def test_landing_death_speed_and_gap(self):
        _,edges,c=self.extract([self.row(-10,0),self.row(0,50),self.row(10,250),self.row(30,350),self.row(40,450)],deaths=[('account.a',self.t+timedelta(seconds=40))])
        self.assertEqual(len(edges),0);self.assertEqual(c['excluded_before_landing'],1)
        self.assertEqual(c['excluded_after_first_death'],1);self.assertEqual(c['rejected_time_or_speed_pairs'],2)
    def test_transport_switch_not_joined_and_vehicle_retained(self):
        _,edges,c=self.extract([self.row(0,50),self.row(10,150,vehicle='Dacia'),self.row(20,650,vehicle='Dacia')])
        self.assertEqual(len(edges),1);self.assertEqual(edges[0,3],1)
        self.assertEqual(c['kind_1_motion_1'],2)


class SplitTests(unittest.TestCase):
    def test_training_aggregation_never_opens_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'match_cache').mkdir()
            np.savez(root/'match_cache/0000.npz',nodes=np.array([[82,1,0,2,1,1,1,1,1,1]]),edges=np.array([[82,83,1,0,1,10,100]]))
            (root/'match_cache/0001.npz').write_text('poison validation file; must not be read')
            rows=[{'match_id':'train','match_index':0,'match_type':'official','split':'train'},{'match_id':'heldout','match_index':1,'match_type':'official','split':'validation'}]
            with patch.object(build,'OUT',root):result=build.aggregate_training(rows)
            self.assertEqual(result[1][82],1);self.assertEqual(result[-1],['train'])
            self.assertEqual(result[5][0][82,83][1],1)


if __name__=='__main__':unittest.main()
