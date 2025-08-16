

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## NOTICE

Copyright (c) 2023-2024, Owners of https://github.com/autogenhub

This project is a fork of https://github.com/microsoft/autogen.

The [original project](https://github.com/microsoft/autogen) is licensed under the MIT License as detailed in [LICENSE_original_MIT](./license_original/LICENSE_original_MIT). The fork was created from version v0.2.35 of the original project.


This project, i.e., https://github.com/autogenhub/autogen, is licensed under the Apache License, Version 2.0 as detailed in [LICENSE](./LICENSE)


Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Ongoing MIT-licensed contributions:
This project regularly incorporates code merged from the [original repository](https://github.com/microsoft/autogen) after the initial fork. This merged code remains under the original MIT license. For specific details on merged commits, please refer to the project's commit history.
The MIT license applies to portions of code originating from the [original repository](https://github.com/microsoft/autogen) as described above.

Last updated: 08/25/2024
"""
logger.info("## NOTICE")

logger.info("\n\n[DONE]", bright=True)