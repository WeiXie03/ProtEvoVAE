'''
Download MSA for the given protein from a protein database
'''

import requests
import json

def download_MSA_pfam(protein_id: str, out_path: str):
    '''
    Download MSA for the given protein from Pfam database
    '''
    url = "https://www.ebi.ac.uk:443/interpro/api/entry/pfam/{0}".format(protein_id)
    req = requests.get(url, params={"annotation": "alignment"})
    with open(out_path, "wb") as outF:
        outF.write(req.content)