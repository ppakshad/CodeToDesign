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

```
C/C++ Source Code
    ↓
[Joern CPG Extraction]
    ↓
[Code Property Graph (CPG)]
    ↓
[Metrics Computation (M1–M16)]
    ↓
[JSON Metadata Export]
    ↓
[LLM Processing (GPT-5)]
    ↓
[UML Use Case Diagram + Security Report]
```

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

