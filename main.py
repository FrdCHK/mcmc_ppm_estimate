"""
description
@Author: Jingdong Zhang
@DATE  : 2024/11/14
"""
import os
import pandas as pd

from mcmc import mcmc


if __name__ == "__main__":
    filenames = [f for f in os.listdir("input") if os.path.isfile(os.path.join("input", f))]
    names_without_extension = [os.path.splitext(f)[0] for f in filenames]
    results = pd.DataFrame(columns=["RA", "RA_ERR", "DEC", "DEC_ERR", "PLX", "PLX_ERR",
                                    "PMRA", "PMRA_ERR", "PMDEC", "PMDEC_ERR",
                                    "RA_DEC_CORR", "RA_PLX_CORR", "RA_PMRA_CORR", "RA_PMDEC_CORR",
                                    "DEC_PLX_CORR", "DEC_PMRA_CORR", "DEC_PMDEC_CORR",
                                    "PLX_PMRA_CORR", "PLX_PMDEC_CORR", "PMRA_PMDEC_CORR"])
    for i in range(len(filenames)):
        data = pd.read_csv(os.path.join("input", filenames[i]))
        result = mcmc(data)
        results = pd.concat([results, result], axis=0, ignore_index=True)
    results.to_csv(os.path.join("output", "example_result.csv"), index=False)
