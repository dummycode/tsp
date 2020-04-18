# TSP Solver
Traveling Salesman Problem Solver

### Authors
Henry Harris \<<hth@gatech.edu>\> <br>
Molly Williams \<<mwilliams401@gatech.edu>\>

### Files
|  File 	|  Description |
|--- |--- |
| tsp.py | Driver file for our TSP solver |
| mat-test.txt | Test input file |
| output-tour.txt | Test output file |
| algorithm.pdf | Our algorithm explained |
| README.md | This markdown file |

### Usage
```
python3 tsp.py <input file> <output file> <time limit>
```

### Bugs
Our solver is running in a separate process that is joined and then terminated after the time has expired. Occasionally (< 5% of executions), this join hangs. Running the solver again fixes this issue.

