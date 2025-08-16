

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# GAIA Benchmark

This scenario implements the [GAIA](https://arxiv.org/abs/2311.12983) agent benchmark. Before you begin, make sure you have followed instruction in `../README.md` to prepare your environment.

### Setup Environment Variables for AgBench

Navigate to GAIA
"""
logger.info("# GAIA Benchmark")

cd benchmarks/GAIA

"""
Update `config.yaml` to point to your model host, as appropriate. The default configuration points to 'gpt-4o'.

Now initialize the tasks.
"""
logger.info("Update `config.yaml` to point to your model host, as appropriate. The default configuration points to 'gpt-4o'.")

python Scripts/init_tasks.py

"""
Note: This will attempt to download GAIA from Hugginface, but this requires authentication.

The resulting folder structure should look like this:

.
./Downloads
./Downloads/GAIA
./Downloads/GAIA/2023
./Downloads/GAIA/2023/test
./Downloads/GAIA/2023/validation
./Scripts
./Templates
./Templates/TeamOne

Then run `Scripts/init_tasks.py` again.

Once the script completes, you should now see a folder in your current directory called `Tasks` that contains one JSONL file per template in `Templates`.

### Running GAIA

Now to run a specific subset of GAIA use:
"""
logger.info("### Running GAIA")

agbench run Tasks/gaia_validation_level_1__MagenticOne.jsonl

"""
You should see the command line print the raw logs that shows the agents in action To see a summary of the results (e.g., task completion rates), in a new terminal run the following:
"""
logger.info("You should see the command line print the raw logs that shows the agents in action To see a summary of the results (e.g., task completion rates), in a new terminal run the following:")

agbench tabulate Results/gaia_validation_level_1__MagenticOne/

"""
## References

**GAIA: a benchmark for General AI Assistants** `<br/>`
Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, Thomas Scialom `<br/>`
[https://arxiv.org/abs/2311.12983](https://arxiv.org/abs/2311.12983)
"""
logger.info("## References")

logger.info("\n\n[DONE]", bright=True)