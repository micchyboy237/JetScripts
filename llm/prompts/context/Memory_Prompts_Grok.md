Dont use memory from previous artifacts.
Execute browse or internet search if requested.
Keep in mind I use a Mac M1.

Use pytest for python tests and test classes to separate functions. Always create result and expected variables for each test. Instead of asserting list lengths, assert list with expected values, Apply this to relevant cases.
I value flexible, modular, testable, optimized, DRY and robust code.

Add debug logs together with the fixes to inspect the latest provided code ONLY after I confirm its not working and give results.
Only provide the final result that doesn't contain the new debug logs after I confirm all tests has passed.
Use types, typed dicts and Literal typing if appropriate.
Only provide the updated parts, functions or tests unless its new or specified otherwise to reduce your generation response.
Do not remove existing function and class definitions. Only update if needed.
Use BDD principles when writing tests. Add "Given", "When", "Then" for each.
Tests should contain complete edge cases and variants of complex input data. This is to improve the code to be more robust on real world scenarios.
If a class or function gets too big, break it down into smaller parts to improve readability without sacrificing logic.
After I confirm all test are working, provide some recommendations that we can still do to improve the code if any.
