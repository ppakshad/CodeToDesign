# RevEngSecure: LLM-Augmented Reverse Engineering for Design-Level Software Defect and Security Analysis

## Abstract

This repository presents **RevEngSecure (Code-to-Design)** — an AI-augmented reverse-engineering framework that bridges source code analysis and design-level security reasoning.  
The system automatically extracts structural and semantic knowledge from **C/C++** projects by constructing a **Code Property Graph (CPG)**, computing sixteen analytical metrics (**M1–M16**), and leveraging **Large Language Models (LLMs)** to regenerate **UML Use Case Diagrams** and behavioral specifications.  
In its final phase, the framework conducts **security compliance analysis** against established standards such as **OWASP**, **CWE**, **STRIDE**, **NIST**, and **ISO**, producing an interpretable and traceable **design-level security report** that maps architectural flaws directly to their implementation evidence.

## Overview

The **Code-to-Design (RevEngSecure)** pipeline performs an end-to-end transformation from raw source code to design-level insight and secure architecture evaluation.  
Through a sequence of automated stages — **static analysis**, **CPG generation**, **metric extraction**, **LLM-driven design recovery**, and **standards-based security verification** — the framework reconstructs accurate high-level documentation for legacy or undocumented systems.  
This enables developers, security analysts, and researchers to visualize software architecture, assess design integrity, and identify latent security weaknesses embedded within code structure and logic.

### Key Features

- **Static Code Analysis**: Uses Joern CPG to extract structural and semantic information from C/C++ projects
- **Comprehensive Metrics Extraction**: Implements 16 distinct metrics (M1–M16) covering functions, control flow, I/O operations, security patterns, and domain semantics
- **AI-Powered Use Case Generation**: Employs GPT-5 to infer use cases, actors, and relationships from code metadata
- **Security Architecture Analysis**: Generates detailed security reports mapping design flaws to OWASP, STRIDE, CWE, and NIST standards
- **PlantUML Diagram Generation**: Automatically produces renderable UML diagrams from extracted use case specifications

## Architecture

The system follows a modular pipeline architecture:

![Proposed Architecture](https://github.com/ppakshad/CodeToDesign/raw/main/Proposed%20Architecture%20(code-to-design)%20(1).png)


### Pipeline Stages

1. **CPG Generation**: Parses C/C++ source code into a Code Property Graph using Joern
2. **Metrics Extraction**: Computes 16 metrics covering structural, semantic, and security aspects
3. **LLM Analysis**: Processes metrics JSON through GPT-5 to generate use case specifications
4. **Diagram Rendering**: Converts PlantUML specifications to visual diagrams
5. **Security Analysis**: Produces comprehensive security architecture reports

## Metrics Framework

The system implements a comprehensive metrics suite (M1–M16) designed to capture evidence for use case extraction:

### Strong Evidence Metrics (Direct UCD Indicators)
- **M2**: Entry Points (main, WinMain, DllMain, public methods)
- **M7**: CLI Arguments Usage (argv/argc occurrences)
- **M8**: I/O Operations (console, file, network, environment)
- **M9**: Name/Text Cues (semantic hints from identifiers and literals)
- **M10**: TF-IDF Domain Terms (bag-of-words and term frequency analysis)
- **M11**: Comments (documentation evidence)
- **M14**: Security Patterns (unsafe calls, high coupling, unvalidated input)
- **M16**: Similarity to UCD Terms (domain vocabulary matching)

### Medium Evidence Metrics (Indirect Support)
- **M3**: Call Graph Edges (function call relationships)
- **M4/M6**: Control Flow Structures (branches, loops, switches)
- **M12**: Complexity Metrics (cyclomatic complexity, recursion, workflow depth)
- **M13**: Cross-Module Calls (inter-module dependencies)
- **M15**: Relations (inheritance, shared globals)

### Weak Evidence Metrics (Scale Indicators)
- **M1**: Function Count (structural size)
- **M5**: CFG Node Sum (control flow graph complexity)


## Metric Queries (Joern / CPG)

This project computes design-relevant metrics (M1–M16) directly from the **Code Property Graph (CPG)** using Joern.  
Below are example queries showing how each metric is derived. You can run these in the Joern shell (`./joern`) or inside a Joern script.

> Note: in the code below, `cpg` is the loaded code property graph and `cleanMethods` is a filtered list of real (non-synthetic) methods.

---

### M1 – Functions (Unique Count)
Count distinct method full names (excluding empty/synthetic ones):

```scala
val cleanMethods = cpg.method.l.filter(_.fullName.nonEmpty)
val m1Pairs = cleanMethods.map { m =>
  val fn = m.fullName.trim
  val sg = Option(m.signature).map(_.trim).getOrElse("")
  (fn, sg)
}.distinct
val M1 = m1Pairs.size
println(s"M1 = $M1")
```


### M2 – Entry Points
Detect classic entry functions (main, WinMain, DllMain) plus public methods:

```scala
val entryPointNames = List("main","WinMain","DllMain")
val entryMethods    = cpg.method.nameExact(entryPointNames:_*).l
val publicMethods   = cpg.method.where(_.modifier.modifierType("PUBLIC")).l

val entryPointFullNames =
  (entryMethods.map(_.fullName) ++ publicMethods.map(_.fullName))
    .map(_.trim)
    .filter(fn => fn.nonEmpty)
    .distinct

val M2 = entryPointFullNames.size
println(s"M2 = $M2")
```

### M3 – Call Graph Edges
Build caller–callee edges from call sites:

```scala
case class Edge(caller:String, callee:String, file:String)

val callEdges = cpg.call.l.map { call =>
  val caller = call.method.fullName
  val callee = Option(call.methodFullName).getOrElse(call.name)
  val file   = call.file.name.headOption.getOrElse("")
  Edge(caller, callee, file)
}.distinct

val M3_callEdgeCount = callEdges.size
println(s"M3_callEdges = $M3_callEdgeCount")
```

### M4 – Branching Structures
Count if, switch, and loop constructs from control structures:

```scala
def lower(s:String) = s.toLowerCase
def isIfCode(s:String) = lower(s).startsWith("if")
def isSwitchCode(s:String) = lower(s).startsWith("switch")
def isLoopCode(s:String) = {
  val t = lower(s)
  t.startsWith("for") || t.startsWith("while") || t.startsWith("do")
}

val m4All = cleanMethods.flatMap { m =>
  m.controlStructure.l.map(_.code.trim)
}

val M4_if     = m4All.count(isIfCode)
val M4_switch = m4All.count(isSwitchCode)
val M4_loop   = m4All.count(isLoopCode)
val M4_total  = m4All.size

println(s"M4_if = $M4_if, M4_switch = $M4_switch, M4_loop = $M4_loop, M4_total = $M4_total")
```

### M5 – CFG Stats
Traverse cfgNext / cfgPrev to measure graph size and depth:

```scala
case class CfgStats(nodes:Int, edges:Int, longest:Int)

def cfgStats(m:Method): CfgStats = {
  val nodes = m.cfgNode.l
  val nodeCount = nodes.size
  val edgeCount = nodes.map(_.cfgNext.size).sum
  val orders = nodes.map(_.order)
  val longest = if (orders.isEmpty) 1 else (orders.max - orders.min + 1)
  CfgStats(nodeCount, edgeCount, longest)
}

val cfgByFunc = cleanMethods.map(m => m.fullName -> cfgStats(m)).toMap

val M5_nodesSum   = cfgByFunc.values.map(_.nodes).sum
val M5_edgesSum   = cfgByFunc.values.map(_.edges).sum
val M5_longestMax = cfgByFunc.values.map(_.longest).foldLeft(0)(math.max)

println(s"M5_cfg_nodes_sum = $M5_nodesSum")
println(s"M5_cfg_edges_sum = $M5_edgesSum")
println(s"M5_cfg_longest_max = $M5_longestMax")
```

### M6 – Global Control Totals
Count all control structures across the whole program:

```scala
val branchCount = cpg.controlStructure.code.l.count(isIfCode)
val loopCount   = cpg.controlStructure.code.l.count(isLoopCode)
val switchCount = cpg.controlStructure.code.l.count(isSwitchCode)

println(s"M6_branch_loop_switch = ($branchCount, $loopCount, $switchCount)")
```


### M7 / M8 – I/O and CLI Usage
Detect CLI arguments and common I/O APIs:

```scala
// CLI
val cliArgsUse = cpg.identifier.nameExact("argv","argc").size
println(s"M7_cliArgs = $cliArgsUse)

// I/O
val ioFile = Set("fopen","fclose","fread","fwrite","open","close","read","write")
val ioNet  = Set("connect","accept","send","recv","socket","listen")
val ioEnv  = Set("getenv","setenv","putenv")

val fileIOCount = cpg.call.name.l.count(ioFile.contains)
val netIOCount  = cpg.call.name.l.count(ioNet.contains)
val envIOCount  = cpg.call.name.l.count(ioEnv.contains)

println(s"M8_file_net_env = ($fileIOCount, $netIOCount, $envIOCount)")
```


### M9 – Name/Text Cues
Match function names and string literals against domain verbs:

```scala
val nameClues = List("login","logout","auth","register","create","delete","update","search")

val nameCueMap =
  cleanMethods.map(_.fullName).distinct.map { fn =>
    val cues = nameClues.filter(k => fn.toLowerCase.contains(k)).toSet
    fn -> cues
  }.toMap

println(s"M9_nameCluesExamples = ${nameCueMap.size}")
```


### M10 – TF–IDF Term Extraction
Builds a bag-of-words per function using identifiers, parameters, and literals, then computes TF–IDF weights to capture domain-relevant tokens.

```scala
import scala.collection.mutable

def splitTokens(s:String): Seq[String] =
  s.toLowerCase.replaceAll("[^a-z0-9_]+"," ").split("\\s+").filter(_.nonEmpty).toSeq

def bagOfWords(m:Method): Seq[String] = {
  val parts = mutable.ArrayBuffer[String]()
  parts += m.name
  parts ++= m.parameter.name.l
  parts ++= m.ast.isIdentifier.name.l
  parts ++= m.ast.isLiteral.code.l
  splitTokens(parts.mkString(" "))
}

val funcBags = cleanMethods.map(m => m.fullName -> bagOfWords(m)).toMap
val termDocFreq = funcBags.values.flatten.distinct.groupBy(identity).view.mapValues(_.size).toMap
val Ndocs = funcBags.size.max(1)

def tfidf(fn:String, term:String): Double = {
  val terms = funcBags.getOrElse(fn, Seq.empty)
  val tf = terms.count(_ == term).toDouble / terms.size.max(1)
  val df = termDocFreq.getOrElse(term, 1).toDouble
  val idf = Math.log((Ndocs + 1.0) / df)
  tf * idf
}

def topTerms(fn:String, k:Int=10): Seq[(String,Double)] =
  funcBags.getOrElse(fn, Seq.empty).distinct.map(t => t -> tfidf(fn,t)).sortBy(-_._2).take(k)

println(s"M10_tfidfTopTermsExamples = ${funcBags.size}")
```


### M11 – Comment Density
Counts comment nodes per function to estimate documentation coverage.

```scala
val commentsPerFunc = cleanMethods.map(m => m.fullName -> m.comment.l.size).toMap
val totalComments = commentsPerFunc.values.sum

println(s"M11_commentsTotal = $totalComments")

```


### M12 – Complexity & Recursion
Approximates cyclomatic complexity and detects recursive functions.

```scala
def cyclomaticApprox(m:Method): Int = {
  val branches = m.controlStructure.l
  1 + branches.count(_.code.startsWith("if")) +
      branches.count(_.code.startsWith("switch")) +
      branches.count(_.code.startsWith("for")) +
      branches.count(_.code.startsWith("while"))
}

def isRecursive(m:Method): Boolean = {
  val fn = m.fullName
  cpg.call.nameExact(fn).nonEmpty
}

case class FlowInfo(cyclo:Int, recursive:Boolean)
val perFuncComplexity = cleanMethods.map(m => m.fullName -> FlowInfo(cyclomaticApprox(m), isRecursive(m))).toMap

val cycloSum = perFuncComplexity.values.map(_.cyclo).sum
val recCount = perFuncComplexity.count(_._2.recursive)

println(s"M12_cyclomaticSum = $cycloSum, recursiveFuncs = $recCount")
```



### M13 – Cross-Module Coupling
Detects inter-file calls and public API exposure to measure modularity:

```scala
case class Edge(caller:String, callee:String, file:String)

val callEdges = cpg.call.l.map { c =>
  Edge(c.method.fullName, Option(c.methodFullName).getOrElse(c.name), c.file.name.headOption.getOrElse(""))
}.distinct

def fileOf(fn:String): String = cpg.method.fullNameExact(fn).file.name.headOption.getOrElse("")

val crossModuleEdges = callEdges.filter(e => fileOf(e.caller) != fileOf(e.callee))
val publicApiList = cpg.method.where(_.modifier.modifierType("PUBLIC")).fullName.l.distinct

println(s"M13_crossModuleEdges = ${crossModuleEdges.size}, publicApiCount = ${publicApiList.size}")
```


### M14 – Security-Sensitive Patterns

Flags unsafe API usage, high-coupling functions, and unvalidated inputs.
```scala
val knownUnsafe = Set("gets","strcpy","strcat","sprintf","scanf","memcpy","memmove")
val unsafeCalls = cpg.call.name.l.filter(knownUnsafe.contains).distinct

val degreeByFn = callEdges.flatMap(e => List(e.caller -> e.callee, e.callee -> e.caller))
  .groupBy(_._1).view.mapValues(_.map(_._2).toSet.size).toMap

val degVals = degreeByFn.values.toSeq.sorted
def percIdx(vs: Seq[Int], p: Double): Int =
  if (vs.isEmpty) 0 else vs((p * (vs.size - 1)).round.toInt.min(vs.size - 1))
val highCouplingThreshold = percIdx(degVals, 0.9)
val highCouplingFuncs = degreeByFn.filter(_._2 >= highCouplingThreshold).keys.toSeq

def unvalidatedInputHeuristic(m:Method): Boolean = {
  val ids = m.ast.isIdentifier.name.l.map(_.toLowerCase)
  val lits = m.ast.isLiteral.code.l.map(_.toLowerCase)
  val hasInput = ids.exists(Set("argv","input","buf","line").contains)
  val hasValidate = (ids ++ lits).exists(s => s.contains("validate") || s.contains("sanitize") || s.contains("check"))
  hasInput && !hasValidate
}

val unvalidatedCount = cleanMethods.count(unvalidatedInputHeuristic)

println(s"M14_unsafeCalls = ${unsafeCalls.size}, highCoupling = ${highCouplingFuncs.size}, unvalidatedInput = $unvalidatedCount")
```


### M15 – Inheritance & Shared Globals
Detects inheritance edges and shared global variables to build class relationships.

```scala
case class InheritEdge(child:String, parent:String)
val inheritanceEdges =
  cpg.typeDecl.l.flatMap(td => td.inheritsFromTypeFullName.l.map(p => InheritEdge(td.fullName, p)))

val allIdentifiers = cpg.identifier.name.l
val globals = allIdentifiers.groupBy(identity).filter(_._2.size >= 2).keys.toList

println(s"M15_inheritanceEdges = ${inheritanceEdges.size}, sharedGlobals = ${globals.size}")
```


### M16 – Use Case Domain (UCD) Term Matching
Matches tokenized code artifacts against provided UCD_TERMS to detect use-case relevance.

```scala
val ucdTerms: Seq[String] =
  sys.env.get("UCD_TERMS")
    .map(_.split(",").toSeq.map(_.trim.toLowerCase))
    .getOrElse(Seq.empty)

val ucdMatches = if (ucdTerms.nonEmpty) {
  funcBags.flatMap { case (fn, terms) =>
    val hits = ucdTerms.filter(t => terms.exists(_.contains(t)))
    if (hits.nonEmpty) Some(fn -> hits) else None
  }
} else Map.empty[String,Seq[String]]

println(s"M16_enabled = ${ucdTerms.nonEmpty}, matchedFunctions = ${ucdMatches.size}")
```



## Installation

### Prerequisites

- **Python 3.8+**
- **Docker** (for Joern CPG extraction)
- **OpenAI API Key** (for GPT-5 access)
- **PlantUML** (optional, for local diagram rendering)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ppakshad/CodeToDesign.git
    cd CodeToDesign
   ```

2. **Install Python Dependencies**
   ```bash
   pip install openai plantuml pathlib
   ```

3. **Configure Docker**
   Ensure Docker is running and can pull the Joern image:
   ```bash
   docker pull ghcr.io/joernio/joern:nightly
   ```

4. **Set OpenAI API Key**
   Create `implementation/openai_api_key.txt` or set the `OPENAI_API_KEY` environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Workflow

1. **Prepare Your C/C++ Project**
   Organize your source code in a directory structure:
   ```
   your_project/
   ├── code/
   │   ├── main.cpp
   │   ├── module1.cpp
   │   └── ...
   └── graphs/  (auto-generated)
   ```

2. **Run the Analysis Pipeline**
   ```bash
   cd implementation
   python project.py
   ```

3. **Provide Project Path**
   When prompted, enter the path to your project's `code` directory:
   ```
   PROJECT path: /path/to/your/project/code
   ```

4. **Review Outputs**
   The pipeline generates:
   - `graphs/usecase_table_metrics_v2.json`: Extracted code metrics
   - `graphs/usecase_usecases.puml`: PlantUML use case diagram
   - `graphs/usecase_usecases.png`: Rendered diagram (if PlantUML server available)
   - `implementation/testout.txt`: Use case bindings with code references
   - `implementation/securityReport.json`: Security architecture analysis

### Advanced Configuration

#### Custom UCD Terms
Set environment variable to provide domain-specific vocabulary:
```bash
export UCD_TERMS="login,logout,register,search,upload,download"
```

#### Model Selection
Override the default GPT-5 model:
```bash
export OPENAI_MODEL="gpt-5-chat-latest"
```

## Project Structure

```
project/
├── implementation/
│   ├── project.py              # Main pipeline implementation
│   ├── docker.sh               # Docker utility script
│   ├── usecaseprompt.txt       # LLM prompt for use case extraction
│   ├── securityprompt.txt      # LLM prompt for security analysis
│   ├── testout.txt             # Generated use case bindings
│   └── securityReport.json     # Generated security report
├── usecase/                    # Analyzed projects (use case diagrams)
│   ├── BankingManagementSystem/
│   ├── LearningManagmentSystem/
│   └── ...
├── spearman/
└── README.md                   
```

## Example Outputs

### Use Case Diagram (PlantUML)
The system generates PlantUML diagrams with:
- **Actors**: Human roles and external systems
- **Use Cases**: User-visible goals derived from code
- **Relationships**: Associations, `<<include>>`, `<<extend>>`, generalizations
- **System Boundary**: Inferred from project structure

### Security Report (JSON)
Comprehensive security analysis including:
- **Design Findings**: Architecture-level security flaws
- **Use Case Bindings**: Code-to-use-case mappings
- **Threat Modeling**: STRIDE-based threat identification
- **Compliance Mapping**: OWASP, CWE, NIST references
- **Remediation Roadmap**: Prioritized action items

## Research Applications

This framework supports research in:
- **Reverse Engineering**: Automated documentation of legacy systems
- **Software Architecture**: Extraction of high-level design from implementation
- **Security Analysis**: Systematic identification of design-level vulnerabilities
- **AI-Assisted Software Engineering**: LLM integration for code understanding
- **Metrics-Driven Development**: Evidence-based use case extraction

## Dataset

The repository includes a curated dataset of 12 C/C++ projects:

| Project | LOC | Files | Functions | Avg Complexity |
|---------|-----|-------|-----------|----------------|
| LearningManagementSystem | 1,427 | 19 | 139 | 1.9 |
| RemoteDesktop | 992 | 11 | 87 | 2.3 |
| ProductManagementTool | 1,015 | 8 | 68 | 2.5 |
| BankingManagementSystem | 3,100 | 1 | 136 | 2.8 |
| HPSocketDev | 29,014 | 53 | 4,027 | 2.2 |
| ... | ... | ... | ... | ... |

**Total**: 38,966 LOC across 121 files, 4,823 functions (weighted avg complexity: 2.22)

## Limitations

- **Language Support**: Currently limited to C/C++ (via Joern CPG)
- **LLM Dependency**: Requires GPT-5 API access (may incur costs)
- **Static Analysis**: Only captures compile-time information (no runtime behavior)
- **Heuristic-Based**: Metrics rely on naming conventions and structural patterns
- **Manual Validation**: Generated diagrams may require domain expert review

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional programming languages
- Enhanced metrics and heuristics
- Improved LLM prompt engineering
- Integration with other static analysis tools
- Performance optimizations for large codebases

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{Pakshad2025CodeToDesign,
  title = {Code-to-Design: LLM-Augmented Reverse Engineering for Design-Level Security},
  author = {Pakshad, Puya},
  year = {2025},
  url = {https://github.com/ppakshad/CodeToDesign}
}
```

## Acknowledgments

- **Joern**: Code Property Graph framework for C/C++
- **OpenAI**: GPT-5 API for semantic analysis
- **PlantUML**: Diagram rendering and visualization

## Contact

For questions, issues, or collaboration inquiries, please open an issue on GitHub or contact [ppakshad@hawk.illinoistech.edu].

---

**Note**: This project is under active development. API interfaces and file structures may change between versions.

