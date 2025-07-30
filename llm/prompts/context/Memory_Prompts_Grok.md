Dont use memory from previous artifacts.
Execute browse or internet search if requested.
Keep in mind I use a Mac M1.

I value flexible, modular, testable, optimized, DRY and robust code.
Do not implement with static code and overly specific defaults.
Write easy to read class or function definitions that you would expect for a newly hired developer to understand quickly.
Use pytest for python tests and test classes to separate behaviors. Always create result and expected variables for each test. Instead of asserting list lengths, assert list with expected values. Use pytest clean up methods if applicable. Apply this to relevant cases.

Add debug logs together with the fixes to inspect the latest provided code ONLY after I confirm its not working and give results.
Only provide the final result that doesn't contain the new debug logs after I confirm all tests has passed.
Use types, typed dicts and Literal typing if appropriate.
Only provide the updated lines, methods or tests unless its new or specified otherwise to reduce your generation response.
Do not remove existing function and class definitions. Only update if needed.
Use BDD principles when writing tests. Add "Given", "When", "Then" for each.
Tests should demonstrate human readable, easy to read real world example inputs and expected variables so I can understand the features better.
Before fixing, analyze provided test results carefully to determine whether the expected variables are correct in logic or if the code needs to be updated.
If a class or function gets too big, break it down into smaller parts to improve readability without sacrificing logic.
After I confirm all test are working, provide some recommendations that we can still do to improve the code if any.
