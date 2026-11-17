"""D2-A SQL only. No E, source attribution, opportunity or regression computation."""
POLICY='1014_APPROVED_HISTORY_POLICY_v2_20261014'
BUILD='D2A_CPC_SEARCH_20261015_v1'
def member_sql(m='target_members',h='target_events',f='target_first'):
 return f"""WITH b AS (SELECT inventor_id,first_filing_date,count(*) code_n FROM {f} GROUP BY 1,2),
 widths AS (SELECT *,sum(code_n) OVER(PARTITION BY inventor_id ORDER BY first_filing_date ROWS UNBOUNDED PRECEDING)::INTEGER prior_cpc_count FROM b),
 batch_gaps AS (SELECT e.inventor_id,e.filing_date,count(*) FILTER(WHERE p.patent_id IS NULL)::BIGINT missing_cpc_patents FROM {h} e LEFT JOIN pc_counts p USING(patent_id) GROUP BY 1,2),
 gaps AS (SELECT inventor_id,filing_date,coalesce(sum(missing_cpc_patents) OVER(PARTITION BY inventor_id ORDER BY filing_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)::BIGINT prior_cpc_missing_patent_count FROM batch_gaps),
 matched AS (SELECT m.patent_id,m.inventor_id,count(p.cpc_subclass)::INTEGER current_cpc_count,
 count(*) FILTER(WHERE p.cpc_subclass IS NOT NULL AND f.first_filing_date<m.filing_date)::INTEGER familiar_cpc_count
 FROM {m} m LEFT JOIN pc p ON p.patent_id=m.patent_id LEFT JOIN {f} f ON f.inventor_id=m.inventor_id AND f.cpc_subclass=p.cpc_subclass GROUP BY 1,2),
 measured AS (SELECT m.*,g.prior_cpc_missing_patent_count,m.prior_patent_count-g.prior_cpc_missing_patent_count prior_cpc_valid_patent_count,
 coalesce(w.prior_cpc_count,0)::INTEGER prior_cpc_count,x.current_cpc_count,x.familiar_cpc_count,x.current_cpc_count-x.familiar_cpc_count new_cpc_count,
 g.prior_cpc_missing_patent_count>0 prior_cpc_gap,
 CASE WHEN m.prior_patent_count>0 THEN g.prior_cpc_missing_patent_count::DOUBLE/m.prior_patent_count END prior_cpc_gap_share
 FROM {m} m JOIN gaps g USING(inventor_id,filing_date) JOIN matched x USING(patent_id,inventor_id)
 ASOF LEFT JOIN widths w ON m.inventor_id=w.inventor_id AND m.filing_date>w.first_filing_date),
 states AS (SELECT *,CASE WHEN history_incomplete_flag THEN 'HISTORY_INCOMPLETE' WHEN prior_patent_count=0 THEN 'FIRST_OBSERVED' WHEN prior_cpc_count>0 THEN 'EXISTING_VALID_HISTORY' ELSE 'PRIOR_WITHOUT_VALID_CPC' END history_state,
 CASE WHEN current_cpc_count>0 AND NOT history_incomplete_flag THEN new_cpc_count::DOUBLE/current_cpc_count END Y_observed,
 CASE WHEN current_cpc_count>0 THEN new_cpc_count::DOUBLE/current_cpc_count END known_history_Y_diagnostic FROM measured),
 flags AS (SELECT *,history_state='EXISTING_VALID_HISTORY' eligible_existing_history,
 history_state='EXISTING_VALID_HISTORY' AND current_cpc_count>0 eligible_existing_Y,
 history_state='EXISTING_VALID_HISTORY' AND current_cpc_count>0 AND NOT prior_cpc_gap strict_cpc_existing_Y_eligible,
 CASE WHEN history_incomplete_flag THEN 'HISTORY_INCOMPLETE' WHEN current_cpc_count=0 THEN 'CURRENT_CPC_EMPTY' WHEN history_state='FIRST_OBSERVED' THEN 'FIRST_OBSERVED_EXCLUDED' WHEN history_state='PRIOR_WITHOUT_VALID_CPC' THEN 'PRIOR_CPC_ALL_EMPTY' ELSE 'DEFINED' END personal_Y_status,
 history_issue_bits+CASE WHEN prior_patent_count>0 AND prior_cpc_count=0 THEN 8 ELSE 0 END+CASE WHEN prior_cpc_gap AND prior_cpc_count>0 THEN 16 ELSE 0 END+CASE WHEN current_cpc_count=0 THEN 32 ELSE 0 END+CASE WHEN career_age_over70 THEN 64 ELSE 0 END search_issue_bits FROM states)
 SELECT * EXCLUDE(cpc_search_eligibility_status,build_version),CASE WHEN eligible_existing_Y THEN Y_observed END personal_Y,
 CASE WHEN strict_cpc_existing_Y_eligible THEN Y_observed END personal_Y_strict,
 '{BUILD}' search_build_version FROM flags"""
def team_sql(m='members',d='d1teams'):
 return f"""WITH a AS (SELECT patent_id,count(*) member_row_count,
 count(*) FILTER(WHERE history_state='FIRST_OBSERVED') first_observed_count,
 count(*) FILTER(WHERE history_state='EXISTING_VALID_HISTORY') existing_valid_history_count,
 count(*) FILTER(WHERE history_state='PRIOR_WITHOUT_VALID_CPC') prior_without_valid_cpc_count,
 count(*) FILTER(WHERE history_state='HISTORY_INCOMPLETE') history_incomplete_count,
 count(*) FILTER(WHERE eligible_existing_Y)::INTEGER m_search,count(*) FILTER(WHERE strict_cpc_existing_Y_eligible)::INTEGER m_search_strict,
 avg(personal_Y) team_existing_Y,avg(personal_Y_strict) team_strict_recomputed_Y,
 avg(Y_observed) known_only_Y_diagnostic,count(Y_observed) known_only_member_count,
 CASE WHEN count(Y_observed)=count(*) THEN avg(Y_observed) END all_member_Y,
 max(current_cpc_count) current_cpc_count,sum(prior_cpc_missing_patent_count)::BIGINT prior_cpc_missing_patent_sum,
 count(*) FILTER(WHERE eligible_existing_Y AND prior_cpc_gap) main_members_with_cpc_gap,
 avg(prior_cpc_gap_share) FILTER(WHERE eligible_existing_Y) main_prior_cpc_gap_share_mean
 FROM {m} GROUP BY 1)
 SELECT d.*,a.* EXCLUDE(patent_id),m_search>0 team_Y_eligible,
 m_search>0 AND m_search=m_search_strict strict_cpc_team_Y_eligible,
 CASE WHEN m_search>0 AND m_search=m_search_strict THEN team_existing_Y END team_strict_fixed_Y,
 m_search_strict>0 strict_recomputed_team_Y_eligible,
 CASE WHEN current_cpc_count=0 THEN 'CURRENT_CPC_EMPTY' WHEN m_search=0 THEN 'NO_ELIGIBLE_EXISTING_MEMBER' ELSE 'DEFINED' END team_Y_status,
 CASE WHEN current_cpc_count=0 THEN 'CURRENT_CPC_EMPTY' WHEN history_incomplete_count>0 THEN 'HISTORY_INCOMPLETE_MEMBER' ELSE 'DEFINED' END all_member_Y_status,
 first_observed_count::DOUBLE/n_full first_observed_share,history_incomplete_count::DOUBLE/n_full history_incomplete_share,
 n_full-m_search n_minus_m_search,
 n_full>=2 AND ai_aipat50 IS NOT NULL AND career_stage_complete AND current_cpc_count>0 AND m_search>0 T2_basic_variable_eligible,
 CASE WHEN all_member_Y IS NOT NULL AND team_existing_Y IS NOT NULL THEN all_member_Y-team_existing_Y END all_minus_existing_Y,
 '{BUILD}' search_build_version FROM {d} d JOIN a USING(patent_id)"""
