import rhita
import os

import pandas as pd

def test_run():
    """
    Test the run function of RHITA.
    """
    
    # Execute the run function
    rhita.run(configfile = 'tests/input/test_config.ini')

    # Check if the output file exists
    out_file = 'tests/output/heatwave/heatwave_test/catalogue.csv'
    assert os.path.exists(out_file)

    # Check the output file has 2 rows
    df = pd.read_csv(out_file)
    assert df.shape[0] == 2