### Testing transformer_1.py

The purpose of this folder is to facilitate testing of updated transformer_1.py scripts. 

To run the tests, copy the updated transformer_1.py to this directory and run test.py from thor:
>> ssh thor 
>> python tests.py

Job IDs will be written to job_ids.txt

Once the jobs are running or completed, compare each subfolder's transformer_1_output.txt to its ./key/transformer_1_output.txt to ensure that the job executed successfully and gave similar results to the original version (which is in ./key/transformer_1_output.txt).