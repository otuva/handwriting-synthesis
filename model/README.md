## Model Training Instructions

In order to train a model, data must be downloaded and placed in this directory.

Follow the download instructions here http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database.

Files required are:

- ascii-all.tar.gz - https://fki.tic.heia-fr.ch/DBs/iamOnDB/data/ascii-all.tar.gz
- lineStrokes-all.tar.gz - https://fki.tic.heia-fr.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz
- original-xml-part.tar.gz - https://fki.tic.heia-fr.ch/DBs/iamOnDB/data/original-xml-part.tar.gz

Only a subset of the downloaded data is required.  Move the relevant download data so the directory structure is as folllows:

```
data/
├── raw/
│   ├── ascii/
│   ├── lineStrokes/
│   ├── original/
|   blacklist.npy
```

Once this is completed, run 

```python
from handwriting_synthesis.training.preparation import prepare
prepare()
```

extract the data and dump it to numpy files.

---

To train the model, run 

```python
from handwriting_synthesis.training import train
train()
```

This takes a couple of days on a single Tesla K80.

