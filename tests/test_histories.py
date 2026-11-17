import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "_shared"))

from study.build import build_all
from study.io import connect


def fixture(folder):
    events=[
        ("p0","1980-01-01","1982-01-01",["S"],["A01A"]),
        ("p1","1990-01-01","1992-01-01",["M"],["C07D"]),
        ("p2","2000-01-01","2000-09-01",["E"],["G06F"]),
        ("p3",None,"1998-01-01",["U"],["B01A"]),
        ("p4","2001-01-01","2002-01-01",["E","S"],["A01A","G06F","G06N"]),
        ("p5","2001-01-01","2002-02-01",["E","S"],["A01A","C07D"]),
        ("p6","2002-01-01","2003-01-01",["E","S"],["A01A","G06F","G06N"]),
        ("p7","2001-01-01","2002-01-01",["U","S"],["B01A","A01A"]),
        ("p8","2001-01-01","2002-01-01",["Z","S"],["B01A","A01A"]),
        ("p9","2003-01-01","2004-01-01",["M"],["C07D"]),
        ("p10","2010-01-01","2012-01-01",["M"],["C07D"]),
        ("p11","2024-01-01","2025-01-01",["M"],["C07D"]),
    ]
    patents=[];members=[];codes=[]
    for pid,filing,grant,people,cpc in events:
        patents.append(dict(patent_id=pid,filing_date=filing,patent_date=grant,patent_type="utility"))
        members.extend(dict(patent_id=pid,inventor_id=i)for i in people)
        codes.extend(dict(patent_id=pid,cpc_subclass=c,cpc_sequence=j)for j,c in enumerate(cpc))
    ai=[dict(patent_id=e[0],aipat_ai_score=.8 if i%2 else.1)for i,e in enumerate(events)]
    locations=[dict(patent_id="p4",latitude=30,longitude=120),dict(patent_id="p4",latitude=40,longitude=-70)]
    citations=[dict(patent_id="p9",citation_patent_id="p4"),dict(patent_id="p10",citation_patent_id="p4")]
    for name,values in[("patents",patents),("inventors",members),("cpc",codes),("ai",ai),("locations",locations),("citations",citations)]:
        pd.DataFrame(values).to_csv(Path(folder)/f"{name}.csv",index=False)
    return dict(data_root=str(folder),inputs={n:f"{n}.csv"for n in["patents","inventors","cpc","ai","locations","citations"]},
                snapshot="2025-12-30",cohort_start=2000,cohort_end=2024,primary_end=2024,
                early_years=5,senior_years=16,ai_threshold=.5,windows=[5,10],threads=1,memory_limit="1GB")


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.cfg=fixture(root);self.out=root/"results";self.out.mkdir()
        self.con=connect(self.out,self.cfg);build_all(self.con,self.cfg,self.out)

    def tearDown(self):
        self.con.close();self.temp.cleanup()

    def get(self,pid,inv):
        return self.con.execute("SELECT * FROM personal WHERE patent_id=? AND inventor_id=?",[pid,inv]).df().iloc[0]

    def test_same_day_patents_do_not_enter_each_others_histories(self):
        early=self.get("p5","E")
        self.assertEqual(early.prior_patents,1)
        self.assertEqual(early.history_width,1)
        self.assertAlmostEqual(early.y,1)
        self.assertAlmostEqual(self.get("p4","E").y,2/3)
        self.assertAlmostEqual(self.get("p6","E").y,0)

    def test_personal_team_and_corpus_scales(self):
        team=self.con.execute("SELECT * FROM teams WHERE patent_id='p4'").df().iloc[0]
        self.assertAlmostEqual(team.relative_y,2/3)
        self.assertAlmostEqual(team.team_y,1/3)
        self.assertEqual(team.pair_n,3)
        self.assertEqual(team.new_pairs,3)
        self.assertAlmostEqual(team.pair_y,1)
        self.assertAlmostEqual(team.mean_age_distance,20,places=6)
        later=self.con.execute("SELECT pair_y FROM teams WHERE patent_id='p6'").fetchone()[0]
        self.assertEqual(later,0)

    def test_career_configuration_and_source_partition(self):
        early,senior=self.get("p4","E"),self.get("p4","S")
        self.assertEqual(early.stage,"E");self.assertEqual(senior.stage,"S")
        self.assertEqual(early.early_peers,0);self.assertEqual(senior.early_peers,1)
        self.assertAlmostEqual(early.B,1/3);self.assertAlmostEqual(early.D,1/3)
        self.assertAlmostEqual(senior.A,1/3);self.assertAlmostEqual(senior.D,1/3)
        self.assertAlmostEqual(senior.A+senior.B+senior.C+senior.D,senior.y)
        self.assertEqual(senior.opportunity_n,1);self.assertEqual(senior.utilization,1)
        self.assertEqual(early.weight,.5)

    def test_unknown_history_is_not_first_and_first_is_not_eligible(self):
        unknown=self.get("p7","U")
        self.assertTrue(unknown.history_unknown);self.assertEqual(unknown.stage,"U")
        self.assertFalse(unknown.is_first);self.assertTrue(pd.isna(unknown.age));self.assertTrue(pd.isna(unknown.y))
        first=self.get("p8","Z")
        self.assertTrue(first.is_first);self.assertFalse(first.existing);self.assertTrue(pd.isna(first.y))

    def test_short_history_windows_and_single_CPC_pair_denominator(self):
        senior=self.get("p4","S")
        self.assertTrue(pd.isna(senior.y5));self.assertTrue(pd.isna(senior.team_y5))
        self.assertAlmostEqual(self.get("p4","E").y5,2/3)
        row=self.con.execute("SELECT pair_n,pair_y FROM teams WHERE patent_id='p2'").fetchone()
        self.assertEqual(row[0],0);self.assertIsNone(row[1])
        # The rolling-window reference must use the same eligible members.
        paired=self.con.execute("SELECT relative_y,relative_full_on_w5,relative_y5,eligible_n,eligible_n5,team_full_on_w5 FROM teams WHERE patent_id='p5'").fetchone()
        self.assertAlmostEqual(paired[0],.75)
        self.assertAlmostEqual(paired[1],1)
        self.assertAlmostEqual(paired[2],1)
        self.assertEqual(paired[3:5],(2,1))
        self.assertIsNone(paired[5])

    def test_geographic_fractional_counts_and_citation_maturity(self):
        mass=self.con.execute("SELECT sum(geographic_weight) FROM geography WHERE patent_id='p4'").fetchone()[0]
        self.assertEqual(mass,1)
        count=self.con.execute("SELECT citations_5y FROM impact WHERE patent_id='p4'").fetchone()[0]
        self.assertEqual(count,1)
        self.assertEqual(self.con.execute("SELECT count(*) FROM impact WHERE patent_id='p11'").fetchone()[0],0)

    def test_parquet_shards_reproduce_the_complete_csv_build(self):
        root=Path(self.temp.name)
        pq=root/"parquet";pq.mkdir()
        columns={}
        with duckdb.connect()as writer:
            for name in["patents","inventors","cpc","ai","locations","citations"]:
                data=pd.read_csv(root/f"{name}.csv")
                parts=[data.iloc[::2],data.iloc[1::2]]if name=="inventors"else[data]
                for i,part in enumerate(parts):
                    writer.register("input_frame",part)
                    writer.sql("SELECT * FROM input_frame").write_parquet(str(pq/f"{name}_{i}.parquet"))
                    writer.unregister("input_frame")
                columns[name]=str(pq/f"{name}_*.parquet")
        cfg=dict(self.cfg,inputs=columns)
        out=root/"parquet_results";out.mkdir()
        with connect(out,cfg)as other:
            build_all(other,cfg,out)
            for table,keys in[("personal","patent_id,inventor_id"),("teams","patent_id"),("impact","patent_id")]:
                expected=self.con.execute(f"SELECT * FROM {table} ORDER BY {keys}").df()
                actual=other.execute(f"SELECT * FROM {table} ORDER BY {keys}").df()
                pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)

    def test_optional_applications_preserve_valid_dates_and_reject_conflicts(self):
        root=Path(self.temp.name)
        app=root/"applications.csv"
        pd.DataFrame(dict(patent_id=["p2"],filing_date=["2000-01-01"])).to_csv(app,index=False)
        cfg=dict(self.cfg,inputs=dict(self.cfg["inputs"],applications=str(app)))
        out=root/"apps_results";out.mkdir()
        with connect(out,cfg)as other:
            build_all(other,cfg,out)
            actual=other.execute("SELECT filing_date,y FROM personal WHERE patent_id='p4'AND inventor_id='E'").fetchone()
            self.assertEqual(str(actual[0]),"2001-01-01")
            self.assertAlmostEqual(actual[1],2/3)
        pd.DataFrame(dict(patent_id=["p2"],filing_date=["1999-01-01"])).to_csv(app,index=False)
        conflict=root/"conflict_results";conflict.mkdir()
        with connect(conflict,cfg)as other:
            with self.assertRaisesRegex(ValueError,"Conflicting filing dates"):
                build_all(other,cfg,conflict)

    def test_forward_citations_use_all_types_in_the_supplied_grant_corpus(self):
        root=Path(self.temp.name)
        p=pd.read_csv(root/"patents.csv")
        p=pd.concat([p,pd.DataFrame([dict(patent_id="design",filing_date="2003-01-01",patent_date="2004-01-01",patent_type="design")])],ignore_index=True)
        p.to_csv(root/"patents.csv",index=False)
        cites=pd.read_csv(root/"citations.csv")
        pd.concat([cites,pd.DataFrame([dict(patent_id="design",citation_patent_id="p4")])],ignore_index=True).to_csv(root/"citations.csv",index=False)
        out=root/"all_types_results";out.mkdir()
        with connect(out,self.cfg)as other:
            build_all(other,self.cfg,out)
            self.assertEqual(other.execute("SELECT count(*) FROM teams WHERE patent_id='design'").fetchone()[0],0)
            self.assertEqual(other.execute("SELECT citations_5y FROM impact WHERE patent_id='p4'").fetchone()[0],2)

    def test_missing_current_CPC_is_not_a_valid_zero_pair_count(self):
        root=Path(self.temp.name)
        codes=pd.read_csv(root/"cpc.csv")
        codes.loc[codes.patent_id!='p11'].to_csv(root/"cpc.csv",index=False)
        out=root/"missing_CPC_results";out.mkdir()
        with connect(out,self.cfg)as other:
            build_all(other,self.cfg,out)
            self.assertEqual(other.execute("SELECT cpc_n FROM teams WHERE patent_id='p11'").fetchone()[0],0)
            row=other.execute("SELECT pair_n,new_pairs,pair_y,pair_any FROM teams WHERE patent_id='p11'").fetchone()
            self.assertEqual(row,(None,None,None,None))
            # A measured singleton CPC retains a valid zero count/occurrence.
            single=other.execute("SELECT pair_n,new_pairs,pair_y,pair_any FROM teams WHERE patent_id='p2'").fetchone()
            self.assertEqual(single,(0,0,None,False))


if __name__=="__main__":unittest.main()
