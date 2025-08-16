

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Packaging AutoGen.NET

This document describes the steps to pack the `AutoGen.NET` project.

## Prerequisites

- .NET SDK

## Create Package

1. **Restore and Build the Project**
"""
logger.info("# Packaging AutoGen.NET")

dotnet restore
dotnet build --configuration Release --no-restore

"""
2. **Create the NuGet Package**
"""
logger.info("2. **Create the NuGet Package**")

dotnet pack --configuration Release --no-build

"""
This will generate both the `.nupkg` file and the `.snupkg` file in the `./artifacts/package/release` directory.

For more details, refer to the [official .NET documentation](https://docs.microsoft.com/en-us/dotnet/core/tools/dotnet-pack).

## Add new project to package list.
By default, when you add a new project to `AutoGen.sln`, it will not be included in the package list. To include the new project in the package, you need to add the following line to the new project's `.csproj` file

e.g.
"""
logger.info("## Add new project to package list.")

<Import Project="$(RepoRoot)/nuget/nuget-package.props" />

"""
The `nuget-packages.props` enables `IsPackable` to `true` for the project, it also sets nenecessary metadata for the package.

For more details, refer to the [NuGet folder](./nuget/README.md).

## Package versioning
The version of the package is defined by `VersionPrefix` and `VersionPrefixForAutoGen0_2` in [MetaInfo.props](./eng/MetaInfo.props). If the name of your project starts with `AutoGen.`, the version will be set to `VersionPrefixForAutoGen0_2`, otherwise it will be set to `VersionPrefix`.
"""
logger.info("## Package versioning")

logger.info("\n\n[DONE]", bright=True)