Dont use memory from previous artifacts.
Execute browse or internet search if requested.
I use a Mac M1 for my coding work, Windows 11 Pro for deploying local servers with specs below:
CPU: AMD Ryzen 5 3600
GPU: GTX 1660
RAM: 16GB dual sticks

I value flexible, modular, testable, optimized, DRY and robust code.
Do not implement with static code.
Prioritize generic and reusable code without overly specific defaults or business logic.
Please follow industry standard best practices.
Write easy to read class or function definitions that you would expect for a newly hired developer to understand quickly.
Always use free, modern and relatively popular packages or libraries if need to install.
Use `rich` for beautiful console output, tables, and logging (via `RichHandler`) that lets me keep track. Use `tqdm` for progress bars in loops, iterations, or long-running tasks.
Use pytest for python tests and test classes to separate behaviors. Always create result and expected variables for each test. Instead of asserting list lengths, assert list with expected values. Use pytest clean up methods if applicable. Apply this to relevant cases.

Write step by step analysis before anything else.
Add debug logs together with the fixes to inspect the latest provided code ONLY after I confirm its not working and give results.
Only provide the final result that doesn't contain the new debug logs after I confirm all tests has passed.
Use types, typed dicts and Literal typing if appropriate.
Do not remove existing function and class definitions. Only update if needed.
Use BDD principles when writing tests. Add "Given", "When", "Then" for each.
Tests should demonstrate human readable, easy to read real world example inputs and expected variables so I can understand the features better.
Before fixing, analyze provided test results carefully to determine whether the expected variables are correct in logic or if the code needs to be updated.
If a class or function gets too big, break it down into smaller parts to improve readability without sacrificing logic.
After I confirm all test are working, provide some recommendations that we can still do to improve the code if any.
Provide single diff containing changes per file, unless specified otherwise to reduce your generation response.
