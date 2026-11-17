import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


class PreparedDataPipeline(unittest.TestCase):
    def test_roster_core_and_windows_with_independent_inputs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            roster = root / "roster"
            roster.mkdir()
            self.write(roster / "part-000.parquet", [
                dict(patent_id=101, history_blob=history, current_blob=b"\x03",
                     history_state="EXISTING_VALID_HISTORY", personal_Y=.5,
                     career_age_years=age, prior_patent_count=1,
                     prior_cpc_count=1, prior_cpc_gap=False)
                for history, age in [(b"\x01", 2.0), (b"\x02", 4.0)]
            ])
            self.write(root / "team.parquet", [dict(
                patent_id=101, filing_date="2010-01-01", grant_date="2011-01-01",
                application_year=2010, ai_aipat50=1, aipat_ai_score=.9,
                n_newcomer=2, n_veteran=0, n_unknown=0, first_observed_count=0,
                ibi_n5v16=0, team_existing_Y=.5, n_full=2)])
            self.write(root / "field.parquet", [dict(patent_id=101, cpc_subclass="A01B")])
            self.write(root / "pairs.parquet", [dict(patent_id=101, K=1, new_count=1,
                                                    new_pairs_exposed_to_undated=0)])
            self.write(root / "members.parquet", [
                dict(patent_id=101, inventor_id=inventor, application_year=2010, n_full=2,
                     E_early_peer_share=1.0, veteran_peer_share=0., first_peer_share=0.,
                     career_age_years=age, prior_patent_count=1, prior_cpc_count=1,
                     personal_Y=.5, career_stage="N", A_share=.5, B_share=0.,
                     C_share=0., D_share=0., early_only_opportunity_n=1,
                     early_only_utilization=1., P1_basic_variable_candidate=True,
                     source_basic_variable_candidate=True, utilization_eligible=True)
                for inventor, age in [(1, 2.), (2, 4.)]
            ])
            self.run_stage(root, "01_data_preparation/build_team_boundaries.py",
                           "--roster-dir", roster, "--output", root / "bounds")
            self.run_stage(root, "03_novelty_indicators/build_core_base.py",
                           "--team-boundaries", root / "bounds/team_boundaries_part-*.parquet",
                           "--team-history", root / "team.parquet", "--primary-cpc", root / "field.parquet",
                           "--pair-history", root / "pairs.parquet", "--personal-candidates", root / "members.parquet",
                           "--output", root / "core")
            history = root / "events.duckdb"
            with duckdb.connect(str(history)) as con:
                con.execute("create table events(patent_id integer, inventor_id integer, filing_date date, shard integer)")
                con.execute("insert into events values (99,1,'2008-01-01',0),(100,2,'2006-01-01',1),(101,1,'2010-01-01',0),(101,2,'2010-01-01',1)")
            self.write(root / "current.parquet", [
                dict(patent_id=p, current_cpc_set=c)
                for p,c in [(99, ["A01B"]), (100, ["A01C"]), (101, ["A01B", "A01C"])]
            ])
            dictionary = root / "dictionary.csv"
            dictionary.write_text("cpc_subclass,code_index\nA01B,0\nA01C,1\n")
            original = root / "core/core.duckdb"
            self.run_stage(root, "02_inventor_histories/extend_history_base.py",
                           "--database", original, "--history-database", history,
                           "--cpc-dictionary", dictionary, "--current-cpc-sets", root / "current.parquet",
                           "--output", root / "windows")
            with duckdb.connect(str(original), read_only=True) as con:
                self.assertEqual(con.execute("select relative_y,team_y,pair_y from teams").fetchone(), (.5, 0., 1.))
                self.assertNotIn("relative_y5", [x[0] for x in con.execute("describe teams").fetchall()])
            with duckdb.connect(str(root / "windows/core.duckdb"), read_only=True) as con:
                self.assertEqual(con.execute("select relative_y5,team_y5,mean_age_distance from teams").fetchone(), (.5, 0., 2.))

    def run_stage(self, directory, script, *arguments):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
            cwd=directory, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @staticmethod
    def write(path, rows):
        pq.write_table(pa.Table.from_pylist(rows), path)
