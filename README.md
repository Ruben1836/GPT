# GPT

## Running the optimizer

`V9_run_optimizer` now requires the paths to the Day Ahead Auction (DAA)
and Intraday Auction (IDA) Excel files to be provided either on the
command line or via an ini file. Example usage:

```bash
python V9_run_optimizer --excel-path-daa path/to/daa.xlsx \
                        --excel-path-ida path/to/ida.xlsx
```

Alternatively place a `V9_run_optimizer.ini` with a `[paths]` section
in the same folder:

```ini
[paths]
excel_path_daa = path/to/daa.xlsx
excel_path_ida = path/to/ida.xlsx
```
