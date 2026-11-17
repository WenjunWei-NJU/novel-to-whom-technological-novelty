"""Build all career and CPC histories directly from patent-level source tables."""
from __future__ import annotations

from pathlib import Path

from .io import export, field, quote, require_unique, source


def ingest(con, cfg):
    p = source(con, "patents", cfg, dict(patent_id="patent_id", filing_date="filing_date", grant_date="patent_date", patent_type="patent_type"))
    application = source(con, "applications", cfg, dict(patent_id="patent_id", filing_date="filing_date"), required=False)
    filing = field(p, "filing_date", "DATE", optional=application is not None)
    con.execute(f"CREATE TABLE patent_input AS SELECT CAST({field(p,'patent_id')} AS VARCHAR) patent_id,{filing} filing_date,{field(p,'grant_date','DATE')} grant_date,{field(p,'patent_type','VARCHAR',optional=True)} patent_type FROM {p['view']}")
    require_unique(con, "patent_input", ["patent_id"])
    if con.execute("SELECT count(*) FROM patent_input WHERE grant_date IS NULL").fetchone()[0]:
        raise ValueError("Missing or invalid grant dates in the granted-patent input")
    if application is not None:
        con.execute(f"CREATE TABLE application_input AS SELECT CAST({field(application,'patent_id')} AS VARCHAR) patent_id,{field(application,'filing_date','DATE')} filing_date FROM {application['view']}")
        require_unique(con, "application_input", ["patent_id"])
        if con.execute("SELECT count(*) FROM patent_input p JOIN application_input a USING(patent_id) WHERE p.filing_date IS NOT NULL AND a.filing_date IS NOT NULL AND p.filing_date<>a.filing_date").fetchone()[0]:
            raise ValueError("Conflicting filing dates in patents and applications inputs")
        con.execute("CREATE TABLE patent_joined AS SELECT p.* EXCLUDE(filing_date),coalesce(a.filing_date,p.filing_date) filing_date FROM patent_input p LEFT JOIN application_input a USING(patent_id)")
    else:
        con.execute("CREATE VIEW patent_joined AS SELECT * FROM patent_input")
    utility = "true" if cfg.get("utility_only_input", False) else "lower(trim(patent_type))='utility'"
    if not cfg.get("utility_only_input", False) and field(p, "patent_type", optional=True) == "NULL":
        raise ValueError("A patent_type column or explicit utility_only_input=true is required")
    con.execute(f"CREATE TABLE patents AS SELECT patent_id,filing_date,grant_date FROM patent_joined WHERE {utility} AND grant_date<=DATE {quote(cfg['snapshot'])}")
    if con.execute("SELECT count(*) FROM patents WHERE filing_date>grant_date").fetchone()[0]:
        raise ValueError("Filing dates after grant dates")
    inv = source(con, "inventors", cfg, dict(patent_id="patent_id", inventor_id="inventor_id"))
    con.execute(f"CREATE TABLE membership AS SELECT CAST({field(inv,'patent_id')} AS VARCHAR) patent_id,CAST({field(inv,'inventor_id')} AS VARCHAR) inventor_id FROM {inv['view']}")
    require_unique(con, "membership", ["patent_id", "inventor_id"])
    if con.execute("SELECT count(*) FROM membership m ANTI JOIN patent_input p USING(patent_id)").fetchone()[0]:
        raise ValueError("Inventor links refer to patents absent from the patent/date input")
    con.execute("DELETE FROM membership WHERE patent_id NOT IN (SELECT patent_id FROM patents)")
    cp = source(con, "cpc", cfg, dict(patent_id="patent_id", cpc="cpc_subclass", sequence="cpc_sequence"))
    if con.execute(f"SELECT count(*) FROM {cp['view']} WHERE {field(cp,'patent_id')} IS NULL OR trim(CAST({field(cp,'patent_id')} AS VARCHAR))=''").fetchone()[0]:
        raise ValueError("Missing patent identifiers in the CPC input")
    con.execute(f"CREATE TABLE cpc AS SELECT CAST({field(cp,'patent_id')} AS VARCHAR) patent_id,upper(trim(CAST({field(cp,'cpc')} AS VARCHAR))) cpc,min({field(cp,'sequence','BIGINT',optional=True,default='0')}) sequence FROM {cp['view']} GROUP BY 1,2")
    con.execute("DELETE FROM cpc WHERE NOT regexp_full_match(cpc,'[A-HY][0-9]{2}[A-Z]') OR cpc IS NULL OR patent_id NOT IN (SELECT patent_id FROM patents)")
    ai = source(con, "ai", cfg, dict(patent_id="patent_id", score="aipat_ai_score"))
    con.execute(f"CREATE TABLE ai AS SELECT CAST({field(ai,'patent_id')} AS VARCHAR) patent_id,{field(ai,'score','DOUBLE')} ai_score FROM {ai['view']}")
    require_unique(con, "ai", ["patent_id"])
    if con.execute("SELECT count(*) FROM ai WHERE ai_score IS NOT NULL AND (NOT isfinite(ai_score) OR ai_score<0 OR ai_score>1)").fetchone()[0]:
        raise ValueError("AI scores outside [0,1]")
    con.execute(f"CREATE TABLE cohort AS SELECT p.*,year(filing_date)::INTEGER application_year,a.ai_score,(a.ai_score>={float(cfg['ai_threshold'])}) ai FROM patents p LEFT JOIN ai a USING(patent_id) WHERE year(filing_date) BETWEEN {int(cfg['cohort_start'])} AND {int(cfg['cohort_end'])}")
    con.execute("CREATE TABLE dictionary AS SELECT cpc,(row_number() OVER(ORDER BY cpc)-1)::INTEGER code FROM (SELECT DISTINCT cpc FROM cpc)")
    n_codes = con.execute("SELECT count(*) FROM dictionary").fetchone()[0]
    if not n_codes:
        raise ValueError("No valid CPC subclasses")
    width = max(8, (n_codes + 7) // 8 * 8)
    zero = f"bitstring('0',{width})"
    con.execute(f"CREATE TABLE patent_cpc AS SELECT c.patent_id,bitstring_agg(d.code,0,{width-1}) bits,count(*)::INTEGER cpc_n,arg_min(c.cpc,struct_pack(s:=c.sequence,c:=c.cpc)) primary_cpc FROM cpc c JOIN dictionary d USING(cpc) GROUP BY 1")
    con.execute("CREATE TABLE unknown_history AS SELECT DISTINCT inventor_id FROM membership JOIN patents USING(patent_id) WHERE filing_date IS NULL")
    issues = source(con, "identity_issues", cfg, dict(inventor_id="inventor_id"), required=False)
    if issues is not None:
        con.execute(f"INSERT INTO unknown_history SELECT DISTINCT CAST({field(issues,'inventor_id')} AS VARCHAR) FROM {issues['view']} EXCEPT SELECT inventor_id FROM unknown_history")
    con.execute(f"CREATE TABLE day_events AS SELECT m.inventor_id,p.filing_date,count(*)::BIGINT day_patents,count(*) FILTER(WHERE c.patent_id IS NULL)::BIGINT day_cpc_gaps,bit_or(coalesce(c.bits,{zero})) bits FROM membership m JOIN patents p USING(patent_id) LEFT JOIN patent_cpc c USING(patent_id) WHERE p.filing_date IS NOT NULL GROUP BY 1,2")
    windows = "".join(
        f",bit_or(bits) OVER(PARTITION BY inventor_id ORDER BY filing_date RANGE BETWEEN INTERVAL '{int(w)} years' PRECEDING AND INTERVAL '1 day' PRECEDING) h{int(w)}"
        for w in cfg["windows"])
    con.execute(f"""CREATE TABLE histories AS SELECT *,
      min(filing_date) OVER(PARTITION BY inventor_id) first_date,
      coalesce(sum(day_patents) OVER(PARTITION BY inventor_id ORDER BY filing_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)::BIGINT prior_patents,
      coalesce(sum(day_cpc_gaps) OVER(PARTITION BY inventor_id ORDER BY filing_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)::BIGINT prior_cpc_gaps,
      bit_or(bits) OVER(PARTITION BY inventor_id ORDER BY filing_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) h{windows}
      FROM day_events""")
    extra = "".join(f",coalesce(h.h{int(w)},{zero}) h{int(w)}" for w in cfg["windows"])
    con.execute(f"""CREATE TABLE member_base AS SELECT m.patent_id,m.inventor_id,p.filing_date,p.grant_date,p.application_year,p.ai,p.ai_score,
      h.first_date,year(h.first_date)::INTEGER entry_cohort,h.prior_patents,h.prior_cpc_gaps,
      u.inventor_id IS NOT NULL history_unknown,
      CASE WHEN u.inventor_id IS NULL THEN date_diff('day',h.first_date,p.filing_date)/365.25 END age,
      coalesce(c.bits,{zero}) current_bits,coalesce(c.cpc_n,0)::INTEGER cpc_n,c.primary_cpc,
      coalesce(h.h,{zero}) h{extra}
      FROM membership m JOIN cohort p USING(patent_id) JOIN histories h ON m.inventor_id=h.inventor_id AND p.filing_date=h.filing_date
      LEFT JOIN unknown_history u ON m.inventor_id=u.inventor_id LEFT JOIN patent_cpc c ON m.patent_id=c.patent_id""")
    con.execute(f"""CREATE TABLE member_stage AS SELECT *,
      CASE WHEN history_unknown THEN 'U' WHEN age<={float(cfg['early_years'])} THEN 'E' WHEN age>={float(cfg['senior_years'])} THEN 'S' ELSE 'M' END stage,
      NOT history_unknown AND prior_patents=0 is_first,
      bit_count(h)::INTEGER history_width,
      NOT history_unknown AND prior_patents>0 AND bit_count(h)>0 AND cpc_n>0 existing,
      NOT history_unknown AND (prior_patents=0 OR bit_count(h)>0) history_usable
      FROM member_base""")
    for table in["member_base","histories","day_events"]:con.execute("DROP TABLE "+table)
    return width


def measure(con, cfg, width):
    zero = f"bitstring('0',{width})"
    windows = [int(w) for w in cfg["windows"]]
    own = "".join(f",CASE WHEN existing AND bit_count(h{w})>0 THEN bit_count(current_bits & ~h{w})::DOUBLE/cpc_n END y{w}" for w in windows)
    con.execute(f"""CREATE TABLE personal_core AS SELECT *,
      CASE WHEN existing THEN bit_count(current_bits & ~h)::DOUBLE/cpc_n END y,
      count(*) OVER(PARTITION BY patent_id)::INTEGER n,
      count(*) FILTER(WHERE stage='E') OVER(PARTITION BY patent_id)::INTEGER n_early,
      count(*) FILTER(WHERE stage='S') OVER(PARTITION BY patent_id)::INTEGER n_senior,
      count(*) FILTER(WHERE stage='U') OVER(PARTITION BY patent_id)::INTEGER n_unknown,
      count(*) FILTER(WHERE is_first) OVER(PARTITION BY patent_id)::INTEGER first_n,
      bool_and(history_usable) OVER(PARTITION BY patent_id) team_history_usable,
      bit_or(h) OVER(PARTITION BY patent_id) team_h{own}
      FROM member_stage""")
    con.execute("DROP TABLE member_stage")
    prefix = "".join(f",bit_or(h{w}) OVER(PARTITION BY patent_id) team_h{w},bool_and(NOT history_unknown AND (prior_patents=0 OR bit_count(h{w})>0)) OVER(PARTITION BY patent_id) team_usable{w}" for w in windows)
    con.execute(f"""CREATE TABLE personal_peers AS SELECT *,
      coalesce(bit_or(CASE WHEN stage='E' THEN h ELSE {zero} END) OVER pre,{zero}) |
      coalesce(bit_or(CASE WHEN stage='E' THEN h ELSE {zero} END) OVER post,{zero}) he,
      coalesce(bit_or(CASE WHEN stage IN ('M','S') THEN h ELSE {zero} END) OVER pre,{zero}) |
      coalesce(bit_or(CASE WHEN stage IN ('M','S') THEN h ELSE {zero} END) OVER post,{zero}) ho{prefix}
      FROM personal_core WINDOW pre AS (PARTITION BY patent_id ORDER BY inventor_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
      post AS (PARTITION BY patent_id ORDER BY inventor_id ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)""")
    con.execute("DROP TABLE personal_core")
    team_windows = "".join(f",CASE WHEN team_usable{w} AND cpc_n>0 THEN bit_count(current_bits & ~team_h{w})::DOUBLE/cpc_n END team_y{w}" for w in windows)
    con.execute(f"""CREATE TABLE personal_sets AS SELECT *,
      CASE WHEN n>=2 AND n_unknown=0 THEN (n_early-(stage='E')::INTEGER)::DOUBLE/(n-1) END early_peers,
      CASE WHEN n>=2 AND n_unknown=0 THEN (n_senior-(stage='S')::INTEGER)::DOUBLE/(n-1) END senior_peers,
      CASE WHEN n>=2 AND n_unknown=0 THEN (first_n-is_first::INTEGER)::DOUBLE/(n-1) END first_peers,
      CASE WHEN existing AND team_history_usable THEN bit_count((current_bits & ~h) & he & ~ho)::DOUBLE/cpc_n END A,
      CASE WHEN existing AND team_history_usable THEN bit_count((current_bits & ~h) & ho & ~he)::DOUBLE/cpc_n END B,
      CASE WHEN existing AND team_history_usable THEN bit_count((current_bits & ~h) & he & ho)::DOUBLE/cpc_n END C,
      CASE WHEN existing AND team_history_usable THEN bit_count((current_bits & ~h) & ~(he | ho))::DOUBLE/cpc_n END D,
      CASE WHEN existing AND team_history_usable THEN bit_count(he & ~ho & ~h)::INTEGER END opportunity_n,
      CASE WHEN existing AND team_history_usable THEN bit_count(current_bits & he & ~ho & ~h)::INTEGER END opportunity_used,
      CASE WHEN bit_count(he | ho)>0 AND team_history_usable THEN bit_count(he & ho)::DOUBLE/bit_count(he | ho) END peer_overlap,
      CASE WHEN team_history_usable AND cpc_n>0 THEN bit_count(current_bits & ~team_h)::DOUBLE/cpc_n END team_y{team_windows}
      FROM personal_peers""")
    con.execute("DROP TABLE personal_peers")
    # Pairwise age distance: each member contributes its rank-weighted age once.
    con.execute("CREATE TABLE age_order AS SELECT patent_id,age,row_number() OVER(PARTITION BY patent_id ORDER BY age,inventor_id) r,n FROM personal_sets WHERE NOT history_unknown")
    con.execute("CREATE TABLE age_distance AS SELECT patent_id,CASE WHEN max(n)>1 THEN sum((2*r-n-1)*age)*2/max(n*(n-1)) END mean_age_distance FROM age_order GROUP BY 1")
    con.execute("DROP TABLE age_order")
    con.execute(f"""CREATE TABLE personal AS SELECT * EXCLUDE(current_bits,h,team_h,he,ho{''.join(',h'+str(w)+',team_h'+str(w) for w in windows)}),
      CASE WHEN opportunity_n>0 THEN opportunity_used::DOUBLE/opportunity_n END utilization,
      CASE WHEN existing AND n>=2 AND n_unknown=0 AND ai IS NOT NULL THEN 1.0/count(*) FILTER(WHERE existing AND n>=2 AND n_unknown=0 AND ai IS NOT NULL) OVER(PARTITION BY patent_id) END weight,
      CASE WHEN existing AND team_history_usable AND n>=2 AND n_unknown=0 AND ai IS NOT NULL THEN 1.0/count(*) FILTER(WHERE existing AND team_history_usable AND n>=2 AND n_unknown=0 AND ai IS NOT NULL) OVER(PARTITION BY patent_id) END source_weight,
      CASE WHEN opportunity_n>0 AND existing AND n>=2 AND n_unknown=0 AND ai IS NOT NULL THEN 1.0/count(*) FILTER(WHERE opportunity_n>0 AND existing AND n>=2 AND n_unknown=0 AND ai IS NOT NULL) OVER(PARTITION BY patent_id) END utilization_weight
      FROM personal_sets""")
    con.execute("DROP TABLE personal_sets")
    window_agg = "".join(
        f",avg(y{w}) relative_y{w}"
        f",avg(y) FILTER(WHERE y{w} IS NOT NULL) relative_full_on_w{w}"
        f",count(*) FILTER(WHERE y{w} IS NOT NULL)::INTEGER eligible_n{w}"
        f",max(team_y{w}) team_y{w}"
        f",max(team_y) FILTER(WHERE team_y{w} IS NOT NULL) team_full_on_w{w}"
        for w in windows)
    con.execute(f"""CREATE TABLE teams AS SELECT patent_id,min(filing_date) filing_date,min(grant_date) grant_date,
      max(application_year) application_year,max(ai)::INTEGER ai,max(ai_score) ai_score,max(primary_cpc) primary_cpc,
      max(n)::INTEGER n,max(n_early)::INTEGER n_early,max(n_senior)::INTEGER n_senior,max(n_unknown)::INTEGER n_unknown,
      max(first_n)::DOUBLE/max(n) first_share,avg(age) age_mean,stddev_pop(age) age_sd,
      avg(ln(1+prior_patents)) logq_mean,avg(history_width) width_mean,avg(y) relative_y,max(team_y) team_y,
      count(*) FILTER(WHERE existing)::INTEGER eligible_n,bool_and(history_usable) history_usable,
      max(prior_cpc_gaps)>0 any_history_gap,
      max(n_early)::DOUBLE/max(n) early_share,max(n_senior)::DOUBLE/max(n) senior_share,
      CASE WHEN max(n)>=2 AND max(n_unknown)=0 THEN 2.0*max(n_early)*max(n_senior)/(max(n)*(max(n)-1)) END ibi,
      max(cpc_n)::INTEGER cpc_n{window_agg}
      FROM personal GROUP BY 1""")
    con.execute("ALTER TABLE teams ADD COLUMN mean_age_distance DOUBLE")
    con.execute("UPDATE teams SET mean_age_distance=CASE WHEN teams.n_unknown=0 THEN a.mean_age_distance END FROM age_distance a WHERE teams.patent_id=a.patent_id")
    con.execute("DROP TABLE age_distance")
    con.execute("ALTER TABLE teams ADD COLUMN peer_covered_y DOUBLE")
    con.execute("UPDATE teams SET peer_covered_y=relative_y-team_y WHERE relative_y IS NOT NULL AND team_y IS NOT NULL")
    if con.execute("SELECT count(*) FROM personal WHERE A IS NOT NULL AND abs(y-A-B-C-D)>1e-10").fetchone()[0]:
        raise ArithmeticError("Personal novelty components do not sum to personal novelty")
    if con.execute("SELECT count(*) FROM teams WHERE team_y>relative_y+1e-10").fetchone()[0]:
        raise ArithmeticError("Team-history novelty exceeds mean personal-history novelty")


def corpus_pairs(con):
    con.execute("""CREATE TABLE pair_history AS SELECT a.cpc cpc_a,b.cpc cpc_b,min(p.filing_date) first_date,
      bool_or(p.filing_date IS NULL) undated FROM cpc a JOIN cpc b ON a.patent_id=b.patent_id AND a.cpc<b.cpc
      JOIN patents p ON a.patent_id=p.patent_id GROUP BY 1,2""")
    con.execute("""CREATE TABLE pair_counts AS SELECT a.patent_id,
      count(*) FILTER(WHERE h.first_date=p.filing_date)::INTEGER new_pairs,
      count(*) FILTER(WHERE h.first_date=p.filing_date AND h.undated)::INTEGER uncertain_new_pairs
      FROM cpc a JOIN cpc b ON a.patent_id=b.patent_id AND a.cpc<b.cpc JOIN cohort p ON a.patent_id=p.patent_id
      JOIN pair_history h ON a.cpc=h.cpc_a AND b.cpc=h.cpc_b GROUP BY 1""")
    con.execute("ALTER TABLE teams ADD COLUMN pair_n BIGINT")
    con.execute("ALTER TABLE teams ADD COLUMN new_pairs BIGINT")
    con.execute("ALTER TABLE teams ADD COLUMN uncertain_new_pairs BIGINT")
    con.execute("ALTER TABLE teams ADD COLUMN pair_y DOUBLE")
    con.execute("UPDATE teams SET pair_n=CASE WHEN cpc_n>0 THEN cpc_n*(cpc_n-1)::BIGINT/2 END,new_pairs=CASE WHEN cpc_n>0 THEN 0 END,uncertain_new_pairs=CASE WHEN cpc_n>0 THEN 0 END")
    con.execute("UPDATE teams SET new_pairs=p.new_pairs,uncertain_new_pairs=p.uncertain_new_pairs FROM pair_counts p WHERE teams.patent_id=p.patent_id")
    con.execute("UPDATE teams SET pair_y=new_pairs::DOUBLE/pair_n WHERE pair_n>0")
    con.execute("ALTER TABLE teams ADD COLUMN pair_any BOOLEAN")
    con.execute("UPDATE teams SET pair_any=(new_pairs>0) WHERE new_pairs IS NOT NULL")


def optional_data(con, cfg):
    geo = source(con, "locations", cfg, dict(patent_id="patent_id", latitude="latitude", longitude="longitude"), required=False)
    if geo is not None:
        con.execute(f"CREATE TABLE location_input AS SELECT DISTINCT CAST({field(geo,'patent_id')} AS VARCHAR) patent_id,{field(geo,'latitude','DOUBLE')} latitude,{field(geo,'longitude','DOUBLE')} longitude FROM {geo['view']}")
        con.execute("DELETE FROM location_input WHERE latitude IS NULL OR longitude IS NULL OR NOT isfinite(latitude) OR NOT isfinite(longitude) OR abs(latitude)>90 OR abs(longitude)>180")
        con.execute("CREATE TABLE geography AS SELECT l.*,1.0/count(*) OVER(PARTITION BY patent_id) geographic_weight,t.ai,t.application_year,t.ibi FROM location_input l JOIN teams t USING(patent_id)")
    cite = source(con, "citations", cfg, dict(citing="patent_id", cited="citation_patent_id"), required=False)
    if cite is not None:
        con.execute(f"CREATE TABLE citations AS SELECT DISTINCT CAST({field(cite,'citing')} AS VARCHAR) citing,CAST({field(cite,'cited')} AS VARCHAR) cited FROM {cite['view']}")
        require_unique(con,"citations",["citing","cited"])
        con.execute(f"""CREATE TABLE impact AS SELECT t.patent_id,count(c.citing)::BIGINT citations_5y
          FROM teams t LEFT JOIN (SELECT c.*,p.grant_date FROM citations c JOIN patent_joined p ON c.citing=p.patent_id WHERE p.grant_date<=DATE {quote(cfg['snapshot'])}) c
          ON t.patent_id=c.cited AND c.grant_date>t.grant_date AND c.grant_date<=t.grant_date+INTERVAL '5 years'
          WHERE t.grant_date<=DATE {quote(cfg['snapshot'])}-INTERVAL '5 years' GROUP BY 1""")


def build_all(con, cfg, out):
    print("Building strict-prior career and CPC histories", flush=True)
    width = ingest(con, cfg)
    measure(con, cfg, width)
    print("Building corpus-first CPC pairs", flush=True)
    corpus_pairs(con)
    optional_data(con, cfg)
    folder = Path(out) / "data"
    export(con, "SELECT * FROM teams ORDER BY patent_id", folder / "teams.parquet")
    export(con, "SELECT * FROM personal ORDER BY patent_id,inventor_id", folder / "inventor_patents.parquet")
    export(con, "SELECT * FROM dictionary ORDER BY code", folder / "cpc_dictionary.csv")
    con.execute("CHECKPOINT")
