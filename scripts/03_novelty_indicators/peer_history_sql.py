"""Native bitset D2B measurement functions; no regression, formal weights or sample freeze."""
POLICY='1014_APPROVED_HISTORY_POLICY_v2_20261014'
BUILD='D2B_EXPOSURE_SOURCE_OA_20261015_v1'

def exposure_sql(m='a_members',t='a_teams'):
 return f"""WITH counts AS (SELECT m.*,t.n_full,t.n_newcomer,t.n_middle,t.n_veteran,t.n_unknown,t.first_observed_count,t.m_search,t.team_existing_Y,t.ibi_n5v16,t.ai_aipat50,
 t.n_newcomer-(m.career_stage='N')::INTEGER early_peer_count,t.n_middle-(m.career_stage='M')::INTEGER middle_peer_count,
 t.n_veteran-(m.career_stage='V')::INTEGER veteran_peer_count,t.n_unknown-(m.career_stage='UNKNOWN')::INTEGER unknown_peer_count,
 t.first_observed_count-(m.history_state='FIRST_OBSERVED')::INTEGER first_peer_count,
 t.history_incomplete_count-(m.history_state='HISTORY_INCOMPLETE')::INTEGER peer_history_incomplete_count,
 t.prior_without_valid_cpc_count-(m.history_state='PRIOR_WITHOUT_VALID_CPC')::INTEGER peer_prior_cpc_empty_count
 FROM {m} m JOIN {t} t USING(patent_id)),flags AS (SELECT *,n_full>=2 AND unknown_peer_count=0 E_descriptive_eligible,
 CASE WHEN n_full=1 THEN 'SINGLETON' WHEN unknown_peer_count>0 THEN 'UNKNOWN_PEER_STAGE' ELSE 'DEFINED' END E_status FROM counts),shares AS (
 SELECT *,CASE WHEN E_descriptive_eligible THEN early_peer_count::DOUBLE/(n_full-1) END E_early_peer_share,
 CASE WHEN E_descriptive_eligible THEN first_peer_count::DOUBLE/(n_full-1) END first_peer_share,
 CASE WHEN E_descriptive_eligible THEN middle_peer_count::DOUBLE/(n_full-1) END middle_peer_share,
 CASE WHEN E_descriptive_eligible THEN veteran_peer_count::DOUBLE/(n_full-1) END veteran_peer_share,
 E_descriptive_eligible AND eligible_existing_Y personal_Y_E_joint_eligible
 FROM flags)
 SELECT *,personal_Y_E_joint_eligible AND ai_aipat50 IS NOT NULL P1_basic_variable_candidate,
 CASE WHEN E_descriptive_eligible THEN (early_peer_count-first_peer_count)::DOUBLE/(n_full-1) END existing_early_peer_share,
 '{BUILD}' exposure_build_version FROM shares"""

def history_sql(m='hm',f='hf',width=680):
 return f"""WITH days AS (SELECT f.inventor_id,f.first_filing_date,bitstring_agg(d.code_index,0,{width-1}) new_bits FROM {f} f JOIN code_dictionary d USING(cpc_subclass) GROUP BY 1,2),prefix AS (
 SELECT inventor_id,first_filing_date,bit_or(new_bits) OVER(PARTITION BY inventor_id ORDER BY first_filing_date ROWS UNBOUNDED PRECEDING) H FROM days)
 SELECT m.patent_id,m.inventor_id,coalesce(p.H,bitstring('0',{width}))::BLOB history_blob,
 bit_count(coalesce(p.H,bitstring('0',{width})))::INTEGER encoded_prior_cpc_count
 FROM {m} m ASOF LEFT JOIN prefix p ON m.inventor_id=p.inventor_id AND m.filing_date>p.first_filing_date"""

def source_sql(e='target_E',h='history_bits',cp='cp_bits',width=680):
 z=f"bitstring('0',{width})"
 return f"""WITH b AS (SELECT e.*,h.history_blob,h.encoded_prior_cpc_count,coalesce(c.current_blob,{z}::BLOB) current_blob,
 h.history_blob::BIT _hi,coalesce(c.current_blob,{z}::BLOB)::BIT _cp,
 count(*) FILTER(WHERE e.prior_cpc_gap) OVER(PARTITION BY e.patent_id)-(e.prior_cpc_gap)::INTEGER peer_cpc_gap_member_count
 FROM {e} e JOIN {h} h USING(patent_id,inventor_id) LEFT JOIN {cp} c USING(patent_id)),w AS (
 SELECT *,bit_or(CASE WHEN career_stage='N' THEN _hi ELSE {z} END) OVER pre _epre,
 bit_or(CASE WHEN career_stage='N' THEN _hi ELSE {z} END) OVER post _epost,
 bit_or(CASE WHEN career_stage IN ('M','V') THEN _hi ELSE {z} END) OVER pre _opre,
 bit_or(CASE WHEN career_stage IN ('M','V') THEN _hi ELSE {z} END) OVER post _opost
 FROM b WINDOW pre AS (PARTITION BY patent_id ORDER BY inventor_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),post AS (PARTITION BY patent_id ORDER BY inventor_id ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)),p AS (
 SELECT *,coalesce(_epre,{z})|coalesce(_epost,{z}) _he,coalesce(_opre,{z})|coalesce(_opost,{z}) _ho,
 personal_Y_E_joint_eligible AND peer_history_incomplete_count=0 AND peer_prior_cpc_empty_count=0 source_main_eligible FROM w),sets AS (
 SELECT *,CASE WHEN source_main_eligible THEN _cp& ~_hi END _j,
 CASE WHEN source_main_eligible THEN _he& ~_ho& ~_hi END _oa,
 source_main_eligible AND first_peer_count=0 source_strict_peer_eligible,
 source_main_eligible AND NOT prior_cpc_gap AND peer_cpc_gap_member_count=0 source_strict_CPC_eligible
 FROM p),categorized AS (
 SELECT *,_j&_he& ~_ho _a,_j&_ho& ~_he _b,_j&_he&_ho _c,_j& ~(_he|_ho) _d FROM sets),numbers AS (
 SELECT *,bit_count(_a)::INTEGER A_count,bit_count(_b)::INTEGER B_count,bit_count(_c)::INTEGER C_count,bit_count(_d)::INTEGER D_count,
 bit_count(_oa)::INTEGER early_only_opportunity_n,bit_count(_cp&_oa)::INTEGER early_only_adopted_n FROM categorized)
 SELECT * EXCLUDE(_hi,_cp,_epre,_epost,_opre,_opost,_he,_ho,_j,_oa,_a,_b,_c,_d),
 CASE WHEN source_main_eligible THEN _he::BLOB END early_peer_history_blob,CASE WHEN source_main_eligible THEN _ho::BLOB END other_peer_history_blob,
 _j::BLOB new_CPC_blob,_a::BLOB A_set_blob,_b::BLOB B_set_blob,_c::BLOB C_set_blob,_d::BLOB D_set_blob,_oa::BLOB opportunity_set_blob,
 A_count::DOUBLE/current_cpc_count A_share,B_count::DOUBLE/current_cpc_count B_share,C_count::DOUBLE/current_cpc_count C_share,D_count::DOUBLE/current_cpc_count D_share,
 CASE WHEN new_cpc_count>0 THEN A_count::DOUBLE/new_cpc_count END A_share_cond_new,
 CASE WHEN new_cpc_count>0 THEN B_count::DOUBLE/new_cpc_count END B_share_cond_new,
 CASE WHEN new_cpc_count>0 THEN C_count::DOUBLE/new_cpc_count END C_share_cond_new,
 CASE WHEN new_cpc_count>0 THEN D_count::DOUBLE/new_cpc_count END D_share_cond_new,
 CASE WHEN early_only_opportunity_n>0 THEN early_only_adopted_n::DOUBLE/early_only_opportunity_n END early_only_utilization,
 source_main_eligible AND early_only_opportunity_n>0 utilization_eligible,
 source_strict_peer_eligible AND source_strict_CPC_eligible source_strict_both_eligible,
 source_main_eligible AND ai_aipat50 IS NOT NULL source_basic_variable_candidate,
 CASE WHEN n_full=1 THEN 'SINGLETON' WHEN NOT eligible_existing_Y THEN 'FOCAL_Y_INVALID' WHEN unknown_peer_count>0 THEN 'UNKNOWN_PEER_STAGE' WHEN peer_history_incomplete_count>0 THEN 'PEER_HISTORY_INCOMPLETE' WHEN peer_prior_cpc_empty_count>0 THEN 'PEER_PRIOR_CPC_EMPTY' ELSE 'DEFINED' END source_status,
 CASE WHEN NOT source_main_eligible THEN 'SOURCE_UNDEFINED' WHEN early_only_opportunity_n=0 THEN 'NO_OPPORTUNITY' ELSE 'DEFINED' END utilization_status,
 (CASE WHEN n_full=1 THEN 1 ELSE 0 END+CASE WHEN NOT eligible_existing_Y THEN 2 ELSE 0 END+CASE WHEN unknown_peer_count>0 THEN 4 ELSE 0 END+CASE WHEN peer_history_incomplete_count>0 THEN 8 ELSE 0 END+CASE WHEN peer_prior_cpc_empty_count>0 THEN 16 ELSE 0 END+CASE WHEN prior_cpc_gap OR peer_cpc_gap_member_count>0 THEN 32 ELSE 0 END+CASE WHEN source_main_eligible AND early_only_opportunity_n=0 THEN 64 ELSE 0 END+CASE WHEN ai_aipat50 IS NULL THEN 128 ELSE 0 END)::INTEGER measurement_issue_bits,
 '{BUILD}' source_build_version FROM numbers"""
