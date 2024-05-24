import pandas as pd
import numpy as np
import sqlalchemy
import pymysql,os
from sqlalchemy import create_engine
from sqlalchemy import text

def get_data(getd):
    engine = create_engine('mysql+pymysql://root:10171996@127.0.0.1:3306/chembl_25', echo=False)
    connection = engine.connect()
    resoverall = connection.execute(getd)
    df2 = pd.DataFrame(resoverall.fetchall())
    #df2.columns = resoverall.keys()
    return df2

def read_sqlf(fname):
    a = pd.read_csv(fname,header=None,sep='\t')
    #print a.shape
    sa = list(a.ix[:,0])
    sb =''
    for im in sa:
        sb= sb+im
    return sb

def write_doc_smi(sqlf,cname):
    #try:
    sqlfa= sqlf.replace('3000',cname)
    ma = get_data(sqlfa)
    smi_name = 'doc_smis/CHEMBL_doc_'+cname+'.smi'
    ma.to_csv(smi_name,header=None,index=None,sep='\t')
    return

if __name__=="__main__":
    sqla = text('SELECT md.molregno AS molregno, md.chembl_id AS chembl_id, cs.canonical_smiles AS canonical_smiles, cs.standard_inchi_key AS standard_inchi_key, cs.standard_inchi AS standard_inchi, cp.acd_most_apka AS acd_most_apka, cp.acd_most_bpka AS acd_most_bpka FROM molecule_dictionary md JOIN compound_structures cs ON md.molregno = cs.molregno JOIN compound_properties cp ON md.molregno = cp.molregno limit 99999999;')

    pka_data = get_data(sqla)
    pka_data.to_csv("chembl_25_acd_pka.tsv", sep="\t", index=None)

