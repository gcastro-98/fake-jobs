import os.path
from typing import List, Dict

"""
Namelist-like file in which the different options and parameters are set
"""

BINARY_NAN_FEATURES: List[str] = ['country', 'state', 'city', 'salary_range']
BOOL_FEATURES: List[str] = ['telecommuting', 'has_company_logo', 'has_questions']
NUMERICAL_FEATURE: str = 'required_doughnuts_comsumption'
SYNTHETIC_FEATURE: str = 'nan_per_sample'

# Short categorical features which will be one-hot encoded. The values of the dict are the categories seen in training
__ONEHOT_FEATURES: Dict[str, List[str]] = {
    'required_experience': ['Entry level', 'nan', 'Mid-Senior level', 'Associate', 'Not Applicable', 'Executive',
                            'Director', 'Internship'],
    'required_education': ["Master's Degree", "Bachelor's Degree", "Professional", "nan", "High School or equivalent",
                           "Unspecified", "Some College Coursework Completed", "Associate Degree", "Vocational",
                           "Certification", "Some High School Coursework", "Vocational - Degree", "Doctorate",
                           "Vocational - HS Diploma"],
    'employment_type': ["Full-time", "Contract", "Part-time", "nan", "Other", "Temporary"]}
ONEHOT_FEATURES: List[str] = list(__ONEHOT_FEATURES.keys())

# Long categorical features which will be embedded in a R^n dimensional space (spaCy + t-SNE transformation)
# Dictionary being the value ``n`` the dimensionality of the space, and the key the ```feature``
__EMBEDDED_FEATURES: Dict[str, int] = {
    'industry': 50, 'function': 50, 'company_profile': 200, 'title': 50,
    'description': 200, 'department': 25, 'requirements': 250, 'benefits': 250}
EMBEDDED_FEATURES: List[str] = list(__EMBEDDED_FEATURES.keys())

# targets to train
LABEL: str = 'fraudulent'

# input files directory
INPUT: str = 'input'
# output files directory
OUTPUT: str = 'output'

for _p in (INPUT, OUTPUT):
    if not os.path.exists(_p):
        os.mkdir(_p)
