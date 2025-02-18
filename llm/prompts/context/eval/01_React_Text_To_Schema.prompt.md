Design a robust JSON schema to structure and evaluate the quality and functionality of AI-generated React.js code. This schema should be highly detailed and adaptable for diverse evaluation scenarios, ensuring comprehensive coverage of all critical aspects of code assessment. Specifically, the schema must include fields to capture evaluations across the following dimensions:

1. **Code Quality**:

   - **Readability**: Assess how easily the code can be read and understood, considering naming conventions, comments, and formatting.
   - **Maintainability**: Evaluate the ease with which the code can be modified or updated.
   - **Scalability**: Determine how well the code handles increasing workloads or requirements.
   - **Adherence to React.js Best Practices**: Confirm whether the code aligns with React.js conventions and standards.

2. **Functionality**:

   - **Correctness of Implementation**: Verify that the code performs as intended according to specifications.
   - **Alignment with Specified Requirements**: Ensure the code meets the stated requirements and expectations.
   - **Handling of Edge Cases**: Check whether edge cases are appropriately addressed in the implementation.

3. **Performance**:

   - **Efficiency of Rendered Components**: Measure how efficiently components are rendered without unnecessary re-renders or performance bottlenecks.
   - **Optimization of Lifecycle Methods**: Evaluate the effective use of React lifecycle methods, hooks, or modern React practices.
   - **Memory and Resource Management**: Assess the code’s efficiency in utilizing memory and system resources.

4. **User Experience (UX)**:

   - **Responsiveness of Components**: Test whether the components are visually responsive across various devices and resolutions.
   - **Accessibility Compliance**: Confirm compliance with accessibility standards, such as ARIA roles, keyboard navigation, and screen reader compatibility.
   - **Integration with Styling Frameworks**: Assess how well the code integrates with CSS frameworks or libraries (e.g., Tailwind CSS, Styled Components).

5. **Testing and Debugging**:

   - **Coverage and Robustness of Unit Tests**: Evaluate the extent and quality of unit tests written for the code.
   - **Error Handling and Logging**: Review mechanisms for catching errors and logging issues effectively.

6. **Meta Information**:
   - **AI-Generated Code Version**: Specify the version of the AI model or code output.
   - **Evaluation Timestamp**: Include the exact timestamp when the evaluation was conducted.
   - **Evaluator’s Credentials**: Identify whether the evaluator is a human or an automated system.

The schema should support hierarchical nesting, enforce data validation rules, and specify which fields are mandatory or optional.

Ensure the schema and results are concise, validated, and consistent with modern JSON schema standards. Structure the schema to facilitate easy parsing and future extensibility.

Output ONLY the JSON evaluation schema wrapped in a code block (use ```json).
