Here are my requirements:

I have a set of instances from the XCSP corpus

We assume all optimization problems are minimization

For ecach instanc :
  - I have the optimal score somewhere
  - For each value in a set of optimality gap values
    - I set the optimality gap value as a bound on the objective variable
    - I solve as a satisfaction problem, for one single solution
    - I construct the search tree from the search trace produced by the solver
    - Foer each node in the seach tree
      - I consider the decision that was made x = v and the decision that was not make x != v
      - I need to know if the corresponding subtree is SAT or UNSAT
        - I know about x=v branch from the search trace
        - For the x!=v branch, I run a solver to see if it is SAT or UNSAT : I define a problem from original instance and partial assignment from search and I solve in a separate process

Each time I solve a problem, or a subproblem I do it in a separate process

Subtree is SAT if it contains a solution of the CSP
Subtree is UNSAT if it contains no solution of the CSP even after exhaustive search

# Storage

I store all information into a Sqlite database

For a problem, a partial assignment is a set of assigment in alphabetical order and is unique as such:

```
x[0,0]!=1
x[0,1]=3
x[0,2]=2
```

For each partial assignment, we keep :
lowest bound (best) found for which it is SAT
highest bound (worst) for which it is UNSAT

This way we can reuse information between solving process, we update the bounds according to the search trees of each resolutions, but info is common for one partial assignment

# Solving

I use minicpbp.jar to solve problems, here is command line interface for now (might require changes e.g. to pass partial assignment)

/home/axel/.jdks/ms-21.0.8/bin/java -javaagent:/home/axel/.local/share/JetBrains/Toolbox/apps/intellij-idea-community-edition/lib/idea_rt.jar=40929 -Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 -Dsun.stderr.encoding=UTF-8 -classpath /home/axel/dev/MiniCPBP-MAP/target/classes:/home/axel/.m2/repository/org/xcsp/xcsp3-tools/2.3/xcsp3-tools-2.3.jar:/home/axel/.m2/repository/javax/json/javax.json-api/1.1.2/javax.json-api-1.1.2.jar:/home/axel/.m2/repository/org/glassfish/javax.json/1.1.2/javax.json-1.1.2.jar:/home/axel/.m2/repository/commons-cli/commons-cli/1.4/commons-cli-1.4.jar:/home/axel/.m2/repository/org/antlr/antlr4/4.7/antlr4-4.7.jar:/home/axel/.m2/repository/org/antlr/antlr4-runtime/4.7/antlr4-runtime-4.7.jar:/home/axel/.m2/repository/org/antlr/antlr-runtime/3.5.2/antlr-runtime-3.5.2.jar:/home/axel/.m2/repository/org/antlr/ST4/4.0.8/ST4-4.0.8.jar:/home/axel/.m2/repository/org/abego/treelayout/org.abego.treelayout.core/1.0.3/org.abego.treelayout.core-1.0.3.jar:/home/axel/.m2/repository/com/ibm/icu/icu4j/58.2/icu4j-58.2.jar launch.SolveXCSPFZN
Missing required options: input, bp-algorithm, branching, search-type, timeout
usage: solve-XCSP
    --bp-algorithm <ALGORITHM>              BP algorithm.
                                            Valid BP algorithms are:
                                            "max-product",
                                            "sum-product"
    --branching <STRATEGY>                  branching strategy.
                                            Valid branching strategies
                                            are:
                                            "dom-wdeg",
                                            "dom-wdeg-max-marginal",
                                            "first-fail-random-value",
                                            "impact-based-search",
                                            "impact-entropy",
                                            "impact-min-entropy",
                                            "max-marginal",
                                            "max-marginal-regret",
                                            "max-marginal-strength",
                                            "min-entropy",
                                            "min-entropy-biased",
                                            "min-entropy-dom-wdeg",
                                            "min-marginal",
                                            "min-marginal-strength",
                                            "min-normalized-entropy",
                                            "min-normalized-entropy-dom-wd
                                            eg"
    --cutoff <CUTOF>                        number of failure before
                                            restart
    --damp-messages                         damp messages
    --damping-factor <LAMBDA>               the damping factor used for
                                            damping the messages
    --dynamic-stop                          BP iterations are stopped
                                            dynamically instead of a fixed
                                            number of iteration
    --entropy-branching-threshold <FLOAT>   entropy branching threshold.
                                            Valid entropy branching
                                            threshold are floats
    --init-impact                           initialize impact before
                                            search
    --input <FILE>                          input FZN or XCSP file
    --max-iter <ITERATIONS>                 maximum number of belief
                                            propagation iterations
    --oracle-on-objective <ORACLE>          oracle on objective.
                                            Valid oracle on objective are
                                            floats
    --propagation-shortcut <BOOL>           propagation shortcut.
                                            Valid propagation shortcut
                                            are:
                                            "False",
                                            "True",
                                            "false",
                                            "true"
    --reset-marginals-before-bp <BOOL>      reset marginals before BP.
                                            Valid reset marginals before
                                            BP are:
                                            "False",
                                            "True",
                                            "false",
                                            "true"
    --restart                               authorized restart during
                                            search (available with dfs
                                            only)
    --restart-factor <restartFactor>        factor to increase number of
                                            failure before restart
    --search-type <SEARCH>                  search type.
                                            Valid search types are:
                                            "dfs",
                                            "lds"
    --skip-uniform-max-prod <BOOL>          skip uniform max product.
                                            Valid skip uniform max prod
                                            are:
                                            "False",
                                            "True",
                                            "false",
                                            "true"
    --solution <FILE>                       file for storing the solution
    --stats <FILE>                          file for storing the
                                            statistics
    --timeout <SECONDS>                     timeout in seconds
    --trace-bp                              trace the belief propagation
                                            progress
    --trace-entropy                         trace the evolution of model's
                                            entropy after each BP
                                            iteration
    --trace-iter                            trace the number of BP
                                            iterations before each
                                            branching
    --trace-search                          trace the search progress
    --var-threshold <variationThreshold>    threshold on entropy's
                                            variation under to stop belief
                                            propagation
    --verify                                check the correctness of
                                            obtained solution

Process finished with exit code 1


# Scheduling solving

I use slurm to schedule solving process
A python scrit generates a batch script that contains all commands in an array an we run it with sbatch and array_id

