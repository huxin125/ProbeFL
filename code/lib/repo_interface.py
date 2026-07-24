from lib.d4j_interface import D4JRepositoryInterface
try:
    from lib.bip_interface import BIPRepositoryInterface
except ImportError:
    BIPRepositoryInterface = None

D4J_PROJECTS = [
    'Chart',
    'Cli',
    'Closure',
    'Codec',
    'Collections',
    'Compress',
    'Csv',
    'Gson',
    'JacksonCore',
    'JacksonDatabind',
    'JacksonXml',
    'Jsoup',
    'JxPath',
    'Lang',
    'Math',
    'Mockito',
    'Time',
]

BIP_PROJECTS = [
    'ansible',
    'cookiecutter',
    'pysnooper',
    'spacy',
    'sanic',
    'httpie',
    'keras',
    'matplotlib',
    'thefuck',
    'pandas',
    'black',
    'scrapy',
    'luigi',
    'fastapi',
    'tornado',
    'tqdm',
    'youtube-dl',
]
    

def get_repo_interface(bug_name, **ri_kwargs):
    def _name_matches_proj_list(name, proj_list):
        return any(name.lower() == proj_name.lower()
                   for proj_name in proj_list)
    proj, bug_num = bug_name.split('_')
    if _name_matches_proj_list(proj, D4J_PROJECTS):
        return D4JRepositoryInterface(bug_name, **ri_kwargs)
    elif _name_matches_proj_list(proj, BIP_PROJECTS):
        return BIPRepositoryInterface(bug_name, **ri_kwargs)
    else:
        raise ValueError(f'Unknown project {proj} detected from {bug_name}.')


def get_bug_metadata(bug_name, **ri_kwargs):
    """[cleaned] bug [cleaned]，[cleaned]"""
    ri = get_repo_interface(bug_name, **ri_kwargs)
    proj, bug_num = bug_name.split('_')

    candidate_method_count = len(ri.method_signatures) if hasattr(ri, 'method_signatures') else 0

    failing_test_count = len(ri.failing_test_signatures) if hasattr(ri, 'failing_test_signatures') else 0

    buggy_method_count = len(ri.buggy_method_signatures) if hasattr(ri, 'buggy_method_signatures') else 0

    available_fail_tests = [
        sig for sig in ri.failing_test_signatures
        if ri.get_test_snippet(sig) is not None
    ] if hasattr(ri, 'failing_test_signatures') else []
    available_test_count = len(available_fail_tests)

    return {
        "project": proj,
        "bug_num": int(bug_num),
        "candidate_method_count": candidate_method_count,
        "failing_test_count": failing_test_count,
        "buggy_method_count": buggy_method_count,
        "available_test_count": available_test_count,
    }