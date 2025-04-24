import os
import re
import random
import pandas as pd
import simplejson as json
from torch.utils.data import Dataset

from trigger.code_trigger import CodeTrigger


class LoadWikiDB:
    def __init__(
        self,
        sqlCreationPath,
        setsPath,
        dbRoot,
        shuffle=False,
    ):
        self.dbRoot = dbRoot
        with open(sqlCreationPath, "r") as js:
            self.sqlCreation = json.load(js)
        with open(setsPath, "r") as js:
            self.sets = json.load(js)

        if shuffle:
            for item in self.sets:
                random.shuffle(item)
        self.maxTableLen = len(self.sets)  # except it

        self.pattern = re.compile(
            r"""
            FOREIGN\s+KEY
            \s*\(\s*"(?P<column>[^"]+)"\s*\)
            \s+REFERENCES\s*
            "(?P<ref_table>[^"]+)"\s*
            \(\s*"(?P<ref_column>[^"]+)"\s*\)
            """,
            re.IGNORECASE | re.VERBOSE,
        )

    def extract_foreign(self, sql_query):
        matches = list(self.pattern.finditer(sql_query))
        results = []
        for m in matches:
            results.append(m.groupdict())
        return results

    def readItem(self, tableSz, itemIdx):
        assert tableSz < self.maxTableLen, "tableSz out of range"
        assert itemIdx < len(self.sets[tableSz]), "itemIdx out of range"

        item = self.sets[tableSz][itemIdx]
        forein_references = {}
        tables = {}
        env = {}
        for tn in item["set"]:
            forein_references[tn] = self.extract_foreign(
                self.sqlCreation[item["database"]][tn]
            )
            tables[tn] = env[tn] = pd.read_csv(
                os.path.join(self.dbRoot, item["database"], "tables", f"{tn}.csv")
            )
        exec("import pandas as pd\nimport numpy as np", env)

        return forein_references, tables, env

    def getMap(self, s, e):
        mapList = []
        for i in range(s, e):
            for j in range(len(self.sets[i])):
                mapList.append((i, j))
        return mapList


prompt_template = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You will be given some tables saved in DataFrame. These tables are from a database and have the following foreign key relations.

```
{foreign}
```

You can write Python codes to explore a database. The following is an example.

```python
{python_code}
```

<Code Result>
{outputs}
</Code Result>

Once you write a Python code block wrapped in '```python```'. It will be executed automatically and the printed results will be give back wrapped in '<Code Result></Code Result>'. Try to use this feature to explore the database and find an interesting conclusion. Your final conclusion should be wrapped in '<Answer></Answer>'.<|im_end|>
<|im_start|>assistant
"""


class WikiDBDataset(Dataset):
    def __init__(
        self,
        sqlCreationPath,
        setsPath,
        dbRoot,
        start,
        end,
        promptTemplate,
    ):
        self.lwd = LoadWikiDB(sqlCreationPath, setsPath, dbRoot)
        self.mapList = self.lwd.getMap(start, end)
        self.promptTemplate = promptTemplate

    def __len__(self):
        return len(self.mapList)

    def __getitem__(self, index):
        realIdx = self.mapList[index]
        forein_references, tables, env = self.lwd.readItem(realIdx[0], realIdx[1])

        foreign = json.dumps(forein_references, indent=2)

        code_list = []
        result_list = []
        for k, v in tables.items():
            code_list.append(f"print('# {k}')\nprint({k}.convert_dtypes().dtypes)")
            result_list.append(f"# {k}\n{v.convert_dtypes().dtypes}")

        prompt = self.promptTemplate.format(
            foreign=foreign,
            python_code="\n".join(code_list),
            outputs="\n".join(result_list),
        )

        # NOTE: The dataset should return a triplet (prompt, trigger, metadata),
        # the metadata is used for reward evaluation.
        return prompt, CodeTrigger(env), None
