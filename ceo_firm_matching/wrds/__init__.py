# WRDS Utilities Subpackage
"""
Centralized WRDS data access and variable construction.
"""
from .connection import connect_wrds, test_wrds_access
from .pulls import (
    pull_boardex_profiles,
    pull_boardex_employment,
    pull_boardex_education,
    pull_boardex_companies,
    pull_ciq_persons,
    pull_ciq_professionals,
    pull_ciq_compensation,
    pull_execucomp_ceos,
    pull_execucomp_full,
    pull_exec_boardex_link,
    pull_boardex_ciq_company_match,
    pull_wrds_people_link,
    pull_ciq_gvkey,
)
from .variables import (
    construct_boardex_variables,
    construct_ciq_variables,
    construct_execucomp_variables,
)

__all__ = [
    # Connection
    "connect_wrds",
    "test_wrds_access",
    # BoardEx Pulls
    "pull_boardex_profiles",
    "pull_boardex_employment",
    "pull_boardex_education",
    "pull_boardex_companies",
    # CIQ Pulls
    "pull_ciq_persons",
    "pull_ciq_professionals",
    "pull_ciq_compensation",
    # ExecuComp Pulls
    "pull_execucomp_ceos",
    "pull_execucomp_full",
    # Crosswalk Pulls
    "pull_exec_boardex_link",
    "pull_boardex_ciq_company_match",
    "pull_wrds_people_link",
    "pull_ciq_gvkey",
    # Variable Construction
    "construct_boardex_variables",
    "construct_ciq_variables",
    "construct_execucomp_variables",
]
