from jet.memory.utils import combine_paths


# Sample data
data = [
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "WORKED_AT",
            {
                "name": "Busybody",
                "position": "Web / Mobile Developer",
                "start_date": {
                    "_Date__ordinal": 739007,
                    "_Date__year": 2024,
                    "_Date__month": 5,
                    "_Date__day": 1
                }
            }
        ]
    },
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "WORKED_AT",
            {
                "end_date": {
                    "_Date__ordinal": 738977,
                    "_Date__year": 2024,
                    "_Date__month": 4,
                    "_Date__day": 1
                },
                "name": "Built Different LLC",
                "position": "Web / Mobile Developer",
                "start_date": {
                    "_Date__ordinal": 738611,
                    "_Date__year": 2023,
                    "_Date__month": 4,
                    "_Date__day": 1
                }
            }
        ]
    },
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "WORKED_AT",
            {
                "end_date": {
                    "_Date__ordinal": 738611,
                    "_Date__year": 2023,
                    "_Date__month": 4,
                    "_Date__day": 1
                },
                "name": "8WeekApp",
                "position": "Web / Mobile Developer",
                "start_date": {
                    "_Date__ordinal": 738307,
                    "_Date__year": 2022,
                    "_Date__month": 6,
                    "_Date__day": 1
                }
            }
        ]
    },
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "WORKED_AT",
            {
                "end_date": {
                    "_Date__ordinal": 737577,
                    "_Date__year": 2020,
                    "_Date__month": 6,
                    "_Date__day": 1
                },
                "name": "Total Assurance Solutions Group",
                "position": "Web / Mobile Developer",
                "start_date": {
                    "_Date__ordinal": 737394,
                    "_Date__year": 2019,
                    "_Date__month": 12,
                    "_Date__day": 1
                }
            }
        ]
    },
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "WORKED_AT",
            {
                "end_date": {
                    "_Date__ordinal": 737394,
                    "_Date__year": 2019,
                    "_Date__month": 12,
                    "_Date__day": 1
                },
                "name": "Zakly Inc.",
                "position": "Web / Mobile Developer",
                "start_date": {
                    "_Date__ordinal": 737060,
                    "_Date__year": 2019,
                    "_Date__month": 1,
                    "_Date__day": 1
                }
            }
        ]
    },
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "WORKED_AT",
            {
                "end_date": {
                    "_Date__ordinal": 737060,
                    "_Date__year": 2019,
                    "_Date__month": 1,
                    "_Date__day": 1
                },
                "name": "ADEC Innovations",
                "position": "Web / Mobile Developer",
                "start_date": {
                    "_Date__ordinal": 736269,
                    "_Date__year": 2016,
                    "_Date__month": 11,
                    "_Date__day": 1
                }
            }
        ]
    },
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "USES",
            {
                "end_date": {
                    "_Date__ordinal": 736208,
                    "_Date__year": 2016,
                    "_Date__month": 9,
                    "_Date__day": 1
                },
                "name": "Asia Pacific Digital",
                "position": "Web / Mobile App Developer",
                "start_date": {
                    "_Date__ordinal": 735538,
                    "_Date__year": 2014,
                    "_Date__month": 11,
                    "_Date__day": 1
                }
            }
        ]
    },
    {
        "path": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "first_name": "Jethro Reuel",
                "gender": "Male",
                "id": "personal-1",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "nationality": "Filipino",
                "preferred_name": "Jethro or Jet"
            },
            "USES",
            {
                "end_date": {
                    "_Date__ordinal": 735538,
                    "_Date__year": 2014,
                    "_Date__month": 11,
                    "_Date__day": 1
                },
                "name": "Entertainment Gateway Group (now Yondu)",
                "position": "Web Developer",
                "start_date": {
                    "_Date__ordinal": 734655,
                    "_Date__year": 2012,
                    "_Date__month": 6,
                    "_Date__day": 1
                }
            }
        ]
    }
]

if __name__ == "__main__":
    from jet.logger import logger
    # Execute function
    result = combine_paths(data)

    # Print result
    import json

    logger.newline()
    logger.debug("RESULT:")
    logger.success(json.dumps(result, indent=2))
